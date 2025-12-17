import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import settings

embeddings_client = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.API_KEY,
    base_url=settings.BASE_URL,
)

if os.path.exists(settings.DB_PATH):
    vectorstore = Chroma(
        persist_directory=settings.DB_PATH,
        embedding_function=embeddings_client,
        collection_name=settings.COLLECTION_NAME
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
else:
    retriever = None

@tool
def search_news(query: str) -> str:
    """
    Поиск информации в базе новостей. 
    Использовать для вопросов о событиях, фактах и темах, требующих фактической проверки.
    """
    if not retriever:
        return "База знаний не найдена. Пожалуйста, запустите ingest.py."
    
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "Информация не найдена."
        
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Ошибка поиска: {e}"