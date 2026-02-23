"""
LangGraph Agent (Latest 2026)
- Correct tool flow
- MemorySaver / Sqlite configurable
- Proper message accumulation
- Compatible with latest LangGraph
"""

import os
from typing import TypedDict, Annotated
from datetime import datetime
from dotenv import load_dotenv

# =========================
# LangChain
# =========================
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# =========================
# LangGraph
# =========================
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

# =========================
# External tools
# =========================
from tavily import TavilyClient
import requests
import sqlite3
load_dotenv()

# ============================================================================
# Tools
# ============================================================================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web using Tavily"""
    try:
        response = tavily_client.search(query, max_results=3)
        results = []
        for r in response.get("results", []):
            results.append(f"{r['title']}\n{r['content']}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get current weather"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]

        return (
            f"Weather in {city}:\n"
            f"Temp: {current['temp_C']}°C\n"
            f"Condition: {current['weatherDesc'][0]['value']}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind: {current['windspeedKmph']} km/h"
        )
    except Exception as e:
        return f"Weather error: {str(e)}"


tools = [search_web, get_weather]

# ============================================================================
# State (LATEST pattern)
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================================
# Model
# ============================================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# Nodes
# ============================================================================

def agent_node(state: AgentState):
    """LLM reasoning node"""

    # Add system message only once at start
    if len(state["messages"]) == 1:
        messages = [
            SystemMessage(
                content="You are a helpful assistant. You can search web and fetch weather."
            )
        ] + state["messages"]
    else:
        messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    # IMPORTANT: return only new message
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]

    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


# ============================================================================
# Graph Builder
# ============================================================================
# def get_sqlite_memory():
#     conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
#     memory = SqliteSaver(conn)
#     return memory
    

def create_agent_graph(memory_type="memory"):
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    workflow.add_edge("tools", "agent")

    # =========================
    # Configurable Memory
    # =========================
    if memory_type == "sqlite":
        conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
        memory = SqliteSaver(conn)
    else:
        memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# ============================================================================
# Runner
# ============================================================================

def run_agent(memory_type):
    agent = create_agent_graph(memory_type)

    config = {
        "configurable": {
            "thread_id": "conversation_2"
        }
    }

    print("\nLangGraph Agent (Latest)")
    print("=" * 50)
    print(f"Memory: {memory_type}")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break

        user_message = HumanMessage(content=user_input)

        for event in agent.stream(
            {"messages": [user_message]},
            config=config,
        ):
            if "agent" in event:
                msg = event["agent"]["messages"][-1]
                if msg.content:
                    print(f"\nAgent: {msg.content}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Options:
    # run_agent("memory")   -> in-memory session
    # run_agent("sqlite")   -> persistent DB
    run_agent()
