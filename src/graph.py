import sqlite3
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings
from src.tools import all_tools

SYSTEM_PROMPT = """Ты — Квен, умный ИИ-ассистент.
Твоя задача — отвечать на вопросы пользователя.
Если предоставлен контекст из инструментов (новости, время), строй ответ ИСКЛЮЧИТЕЛЬНО на нём.
Если контекст говорит "Информация не найдена" или "Не релевантно", так и скажи пользователю, не выдумывай факты.
"""

llm = ChatOpenAI(
    api_key=settings.API_KEY,
    base_url=settings.BASE_URL,
    model=settings.MODEL_NAME,
    temperature=0
)

llm_with_tools = llm.bind_tools(all_tools)

class Grade(BaseModel):
    """Оценка релевантности полученного документа."""
    binary_score: str = Field(description="Релевантен ли документ запросу? 'yes' или 'no'")

def node_grader(state):
    """
    Узел, который проверяет последний ToolMessage на соответствие вопросу пользователя.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if not isinstance(last_message, ToolMessage):
        return {"messages": []}

    question = "Неизвестный вопрос"
    for msg in reversed(messages[:-1]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    context = last_message.content

    structured_llm_grader = llm.with_structured_output(Grade)

    system_msg = SystemMessage(content="""Ты — строгий оценщик релевантности. 
    Твоя задача — проверить, содержит ли документ ответ на вопрос пользователя или связанную информацию.
    Отвечай только 'yes' или 'no'.""")
    
    grade_prompt = ChatPromptTemplate.from_messages([
        system_msg,
        ("human", "Вопрос пользователя: {question}\n\nДокумент из базы:\n{context}")
    ])
    
    grader_chain = grade_prompt | structured_llm_grader
    
    try:
        scored_result = grader_chain.invoke({"question": question, "context": context})
        score = scored_result.binary_score
    except Exception as e:
        print(f"Ошибка грейдера: {e}")
        score = "yes"

    if score.lower() == "yes":
        print("--- GRADER: Документ релевантен ---")
        return {"messages": []}
    else:
        print("--- GRADER: Документ НЕ релевантен (фильтрация) ---")
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_call_id,
                    content="В базе знаний найдена информация, но она не соответствует вашему запросу. Игнорируй этот контекст."
                )
            ]
        }

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State):
    history = state["messages"]
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + history
    response = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [response]}

tool_node = ToolNode(all_tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("grader", node_grader)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "grader")

graph_builder.add_edge("grader", "chatbot")

conn = sqlite3.connect("history.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

app = graph_builder.compile(checkpointer=memory)

def save_graph_image():
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("graph_visualization.png", "wb") as f:
            f.write(png_data)
        print("Граф сохранен в graph_visualization.png")
    except Exception as e:
        print(f"Не удалось нарисовать граф: {e}")