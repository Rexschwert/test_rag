import sys
import uuid
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph import app, save_graph_image

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"--- Квен запущен (Session ID: {thread_id[:8]}) ---")
    print("Команды: 'exit' - выход, 'graph' - сохранить схему бота.")

    while True:
        try:
            user_input = input("\nВы: ")
            if not user_input.strip(): continue
            
            if user_input.lower() in ["exit", "выход"]:
                print("Квен: До связи!")
                break
            
            if user_input.lower() == "graph":
                save_graph_image()
                continue

            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="values"
            )
            
            for event in events:
                messages = event.get("messages")
                if not messages:
                    continue
                
                last_msg = messages[-1]
                
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        print(f"\033[93mИспользую: {tool_name} с аргументами {tool_args}...\033[0m")

                elif isinstance(last_msg, ToolMessage):
                    snippet = last_msg.content[:100] + "..." if len(last_msg.content) > 100 else last_msg.content
                    print(f"\033[90mИнструмент вернул данные: {snippet}\033[0m")

                elif isinstance(last_msg, AIMessage) and last_msg.content and not last_msg.tool_calls:
                    print(f"\nКвен: {last_msg.content}")

        except KeyboardInterrupt:
            print("\nКвен: Выход...")
            break
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            print("Произошла ошибка. Попробуйте другой вопрос.")

if __name__ == "__main__":
    main()