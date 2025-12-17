import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("GPUSTACK_API_KEY")
    BASE_URL = os.getenv("GPUSTACK_BASE_URL")
    MODEL_NAME = os.getenv("LLM_MODEL_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DB_PATH = os.path.join(BASE_DIR, os.getenv("CHROMA_DB_DIR", "chroma_db"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "lenta_news")
    
    DATA_FILE = os.path.join(BASE_DIR, "lenta-ru-news.csv")
    
    INGEST_LIMIT = 100 
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200

settings = Config()