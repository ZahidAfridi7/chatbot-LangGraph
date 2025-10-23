from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated,TypedDict,List
from dotenv import load_dotenv
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI()

class ChatState(TypedDict):
    message : Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState):
    pass

graph = StateGraph(ChatState)


graph.add_node('chat_node', chat_node)
graph.add_node()

