from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):
    prompt = f'generate a joke on the topic {state["topic"]}'
    response = model.invoke(prompt).content
    return {'joke': response}

def generate_explanation(state: JokeState):
    prompt = f'write an explanation for the joke {state["joke"]}'
    response = model.invoke(prompt).content
    return {'explanation': response}

graph = StateGraph(JokeState)
graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = "1"
while True:
    user_message = input("type_here: ")
    print("User:", user_message)
    if user_message.strip().lower() in ["exit", "quit", "bye"]:
        break

    # Corrected the typo from 'configruable' to 'configurable'
    config = {'configurable': {'thread_id': thread_id}}

    result = chatbot.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
    print("AI:", result["messages"][-1].content)
