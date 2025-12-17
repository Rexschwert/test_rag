import os
import warnings
import pandas as pd
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

warnings.filterwarnings('ignore')

def process_row(row):
    """Форматирует одну строку датафрейма в заголовок для контекста."""
    return (
        f"Название: {str(row.get('title', 'Без заголовка'))} | "
        f"Тема: {str(row.get('topic', 'Нет'))} | "
        f"Тэги: {str(row.get('tags', 'Нет'))} | "
        f"Дата: {str(row.get('date', 'Нет'))} | "
        f"Ссылка: {str(row.get('url', 'Нет'))} | "
    )

def ingest_data():
    print(f"--- Запуск инджеста ---")
    print(f"Файл данных: {settings.DATA_FILE}")
    print(f"База данных: {settings.DB_PATH}")

    if not os.path.exists(settings.DATA_FILE):
        print(f"!!! ОШИБКА: Файл {settings.DATA_FILE} не найден. Положите файл в корень проекта.")
        return

    embeddings_client = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        chunk_size=16 
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    print("Чтение и обработка CSV...")
    all_documents = []
    processed_count = 0
    
    df_iterator = pd.read_csv(
        settings.DATA_FILE, 
        usecols=['url', 'title', 'text', 'topic', 'tags', 'date'], 
        chunksize=500
    )

    with tqdm(total=settings.INGEST_LIMIT, desc="Обработка статей") as pbar:
        for chunk_df in df_iterator:
            chunk_df.dropna(subset=['text'], inplace=True)
            
            for _, row in chunk_df.iterrows():
                if processed_count >= settings.INGEST_LIMIT:
                    break
                
                header = process_row(row)
                text_to_split = str(row['text'])
                text_chunks = text_splitter.split_text(text_to_split)
                
                for chunk in text_chunks:
                    final_content = header + chunk
                    
                    metadata = {
                        "source": str(row.get('url', 'Нет')),
                        "date": str(row.get('date', 'Нет')),
                        "title": str(row.get('title', 'Без заголовка'))
                    }
                    
                    all_documents.append(
                        Document(page_content=final_content, metadata=metadata)
                    )
                
                processed_count += 1
                pbar.update(1)

            if processed_count >= settings.INGEST_LIMIT:
                break

    print(f"\nОбработано статей: {processed_count}")
    print(f"Сформировано чанков: {len(all_documents)}")

    if all_documents:
        print(f"Подключение к ChromaDB по пути '{settings.DB_PATH}'...")
        vectorstore = Chroma(
            persist_directory=settings.DB_PATH,
            embedding_function=embeddings_client,
            collection_name=settings.COLLECTION_NAME
        )
        
        batch_size = 500
        print(f"Загрузка эмбеддингов...")
        with tqdm(total=len(all_documents), desc="Индексация") as pbar:
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                vectorstore.add_documents(documents=batch)
                pbar.update(len(batch))
                
        print(f"\nБаза данных обновлена")
        print(f"Всего чанков: {vectorstore._collection.count()}")
    else:
        print("Нет данных для загрузки")

if __name__ == "__main__":
    ingest_data()