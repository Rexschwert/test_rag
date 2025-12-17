import datetime
from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """
    Возвращает текущие дату и время в формате ISO.
    Используй этот инструмент, когда пользователь спрашивает "который час?", "какое сегодня число?",
    "какое сейчас время?" или похожие вопросы о текущем времени/дате.
    """
    return datetime.datetime.now().isoformat()