from pydantic import BaseModel
from typing import Callable
from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool

from langgraph.store.memory import InMemoryStore

load_dotenv()

#______Tools_________________
@tool
def search_tool(query: str) -> str:
    """Search the web"""
    return f"Search result: {query}"

@tool
def analysis_tool(query: str) -> str:
    """Analyse data"""
    return f"Analysis data: {query}"

@tool
def export_tool(query: str) -> str:
    """Export the data from file"""
    return f"Export data: {query}"

#_____Context__________________
class Context(BaseModel):
    user_id: str

#______Middleware_________________
@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)
    else:
        request = request.override(tools=[])

    print(f"Tools avaiable : {request.tools}")

    return handler(request)

memory_store = InMemoryStore()

memory_store.put(
    ("features",),
    "user_001",
    {"enabled_tools": ["search_tool"]}
)

memory_store.put(
    ("features",),
    "user_002",
    {"enabled_tools": ["search_tool", "analysis_tool", "export_tool"]}
)

agent = create_agent(
    model="google_genai:gemini-3-flash-preview",
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=memory_store
)

# ── Run for User 1 (basic) ─────────────────────────
print("=" * 40)
print("USER 1 — Basic Plan")
print("=" * 40)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What tools do I have?  1) search_tool 2) analysis_tool 3) export_tool"}]},
    context=Context(user_id="user_001")
)

print(result["messages"][-1].content[-1]["text"])
# ── Run for User 2 (premium) ──────────────────────
print("=" * 40)
print("USER 2 — Premium Plan")
print("=" * 40)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What tools do I have? 1) search_tool 2) analysis_tool 3) export_tool"}]},
    context=Context(user_id="user_002")
)
print(result["messages"][-1].content[-1]["text"])