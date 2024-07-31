import os
import requests
from tqdm import tqdm
import rdflib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fire
from llama_cpp import Llama
from telegram import Update, Voice
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from vosk import Model, KaldiRecognizer
import wave
import json
import pyaudio
import zipfile

# Константы
LLAMA_MODEL_URL = "https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/model-q2_K.gguf"
LLAMA_MODEL_PATH = "model-q2_K.gguf"
ONTOLOGY_URL = "https://dbpedia.org/data/Leonardo_da_Vinci.ttl"
TELEGRAM_BOT_TOKEN = " " #Вставить сюда токен для вашего тг-бота

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
VOSK_MODEL_DIR = "model"
VOSK_MODEL_PATH = os.path.join(VOSK_MODEL_DIR, "vosk-model-small-ru-0.22")

SYSTEM_PROMPT = "Ты преподователь университета, твоя цель отвечать на вопросы о Леонардо да Винчи"
SYSTEM_TOKEN = 1788
USER_TOKEN = 1404
BOT_TOKEN = 9225
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

# Функция для загрузки LLM
def download_model():
    if os.path.exists(LLAMA_MODEL_PATH):
        print(f"Model already exists at {LLAMA_MODEL_PATH}. Skipping download.")
        return
    
    response = requests.get(LLAMA_MODEL_URL, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(LLAMA_MODEL_PATH, 'wb') as f, tqdm(
        desc=LLAMA_MODEL_PATH,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
    
    print(f"Model downloaded and saved to {LLAMA_MODEL_PATH}")

# Функция загрузки и распаковки STT
def download_and_extract_vosk_model(url, output_dir):
    if os.path.exists(output_dir):
        print(f"VOSK model already exists at {output_dir}. Skipping download.")
        return
    
    zip_path = output_dir + ".zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f, tqdm(
        desc=zip_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(zip_path)
    print(f"VOSK model downloaded and extracted to {output_dir}")

# Загрузка онтологии
def load_ontology():
    g = rdflib.Graph()
    g.parse(ONTOLOGY_URL)
    return g

# Извлечение текстов из онтологии
def extract_texts(graph):
    texts = []
    for s, p, o in graph:
        if isinstance(o, rdflib.Literal):
            texts.append(str(o))
    return texts

# Создание эмбеддингов и векторного хранилища
def create_embeddings_and_index(texts):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, 'faiss_index')
    np.save('texts.npy', np.array(texts))

    return model, index

# Функция для поиска ближайших текстов в векторном хранилище
def search_texts(query, index, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[idx] for idx in indices[0]]
    return results

# Класс для генерации ответов с использованием llamы(сайги)
class RAGSystem:
    def __init__(self, index, embedding_model, texts, llama_model, llama_tokenizer):
        self.index = index
        self.embedding_model = embedding_model
        self.texts = texts
        self.llama_model = llama_model
        self.llama_tokenizer = llama_tokenizer

    def get_message_tokens(self, model, role, content):
        message_tokens = model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, ROLE_TOKENS[role])
        message_tokens.insert(2, LINEBREAK_TOKEN)
        message_tokens.append(model.token_eos())
        return message_tokens

    def get_system_tokens(self, model):
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
        return self.get_message_tokens(model, **system_message)

    def generate_response(self, query):
        results = search_texts(query, self.index, self.embedding_model, self.texts)
        context = " ".join(results)
        
        message_tokens = self.get_message_tokens(self.llama_model, role="user", content=query)
        system_tokens = self.get_system_tokens(self.llama_model)
        tokens = system_tokens + message_tokens
        role_tokens = [self.llama_model.token_bos(), BOT_TOKEN, LINEBREAK_TOKEN]
        tokens += role_tokens

        generator = self.llama_model.generate(
            tokens,
            top_k=30,
            top_p=0.9,
            temp=0.2,
            repeat_penalty=1.1
        )
        
        response = ""
        for token in generator:
            token_str = self.llama_model.detokenize([token]).decode("utf-8", errors="ignore")
            tokens.append(token)
            if token == self.llama_model.token_eos():
                break
            response += token_str
        
        return response

# Функции для работы с tg
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Задайте любой вопрос о Леонардо да Винчи. Можно использовать как текст, так и голосовые сообщения')

def handle_message(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    response = rag_system.generate_response(user_message)
    update.message.reply_text(response)

def handle_voice_message(update: Update, context: CallbackContext) -> None:
    voice = update.message.voice.get_file()
    voice.download("voice.ogg")
    
    # Конвертируем ogg в wav
    os.system("ffmpeg -i voice.ogg -y voice.wav")
    
    # Распознаем речь
    wf = wave.open("voice.wav", "rb")
    if wf.getnchannels() != 1:
        raise ValueError("Audio file should be single channel (mono)")
    if wf.getsampwidth() != 2:
        raise ValueError("Audio file should be 16-bit")
    if wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
        raise ValueError("Sample rate should be one of [8000, 16000, 32000, 44100, 48000]")

    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            break

    result = json.loads(rec.FinalResult())
    text = result.get("text", "")
    
    response = rag_system.generate_response(text)
    update.message.reply_text(response)
    
    # Удаляем временные файлы
    os.remove("voice.ogg")
    os.remove("voice.wav")

def main(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1
):
    download_model()
    download_and_extract_vosk_model(VOSK_MODEL_URL, VOSK_MODEL_DIR)

    graph = load_ontology()
    texts = extract_texts(graph)

    embedding_model, index = create_embeddings_and_index(texts)

    # Загрузка модели
    llama_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
    )
    llama_tokenizer = llama_model.tokenizer

    global rag_system
    rag_system = RAGSystem(index, embedding_model, texts, llama_model, llama_tokenizer)

    # Настройка и запуск Telegram-бота
    updater = Updater(TELEGRAM_BOT_TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(MessageHandler(Filters.voice & ~Filters.command, handle_voice_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    fire.Fire(main)
