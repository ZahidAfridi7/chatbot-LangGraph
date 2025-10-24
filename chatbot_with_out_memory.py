import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")  

# Define the chat state type
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Function that handles chatbot node
def chat_node(state: ChatState):
    messages = state["messages"]
    if not messages:
        raise ValueError("state['messages'] must contain at least one message")
    response = model.invoke(messages)  # returns an AIMessage
    return {"messages": [response]}  # MUST be a list, not a set

# Setup memory and graph
checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def get_chat_response(user_input: str, thread_id: str = "1"):
    """This function processes the user input and returns the chatbot's response."""
    
    # Prepare the state (messages history)
    state = {'messages': [HumanMessage(content=user_input)]}
    
    config = {'configurable': {'thread_id': thread_id}}

    try:
        # Invoke the chatbot with the user input and the state
        result = chatbot.invoke(state, config=config)
        ai_message = result["messages"][-1].content
        return ai_message
    except Exception as e:
        return f"Error: {e}"

