import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph import app

st.set_page_config(page_title="Qwen RAG Agent", page_icon="🤖")
st.title("Демо RAG-агент")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if prompt := st.chat_input("Введите ваш вопрос"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        status_placeholder = st.status("Думаю...", expanded=True)
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        try:
            events = app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config,
                stream_mode="values"
            )
            
            for event in events:
                messages = event.get("messages")
                if not messages: continue
                last_msg = messages[-1]

                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]['name']
                    status_placeholder.write(f"Вызываю инструмент: **{tool_name}**")
                
                elif isinstance(last_msg, ToolMessage):
                    status_placeholder.write(f"Получены данные. Проверка релевантности...")

                elif isinstance(last_msg, AIMessage) and last_msg.content:
                    full_response = last_msg.content
                    message_placeholder.markdown(full_response)
            
            status_placeholder.update(label="Готово!", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")