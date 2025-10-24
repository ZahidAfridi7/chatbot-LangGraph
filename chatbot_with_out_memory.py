from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")  

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    if not messages:
       
        raise ValueError("state['messages'] must contain at least one message")
    response = model.invoke(messages)          # returns an AIMessage
    return {"messages": [response]}            # MUST be a list, not a set


checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "1"
while True:
    user_message = input("type_here: ")
    print("User:", user_message)
    if user_message.strip().lower() in ["exit", "quit", "bye"]:
        break

    config = {'configruable': {'thread_id': thread_id}}

    result = chatbot.invoke({"messages": [HumanMessage(content=user_message)]},config=config)
    print("AI:", result["messages"][-1].content)
