from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

from typing import Callable

from dotenv import load_dotenv

load_dotenv()

#_______________Defining Tools________
@tool
def public_search(query: str) -> str:
    """Search public information."""
    return f"Public result for: {query}"

@tool
def private_search(query: str) -> str:
    """Search private information."""
    return f"Private result for: {query}"

@tool
def advanced_search(query: str) -> str:
    """Search advanced information."""
    return f"Advanced result for: {query}"


#_______________Middleware__________
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    
    print(f"Before tools", request.tools)
    state = request.state
    is_authenticated = state.get("authenticated", False)
    print(f"Authenticated : {is_authenticated}")
    message_count = len(state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
    elif message_count < 5:
        tools = [t for t in request.tools if t.name.startswith("advanced_")]
    else:
        tools = request.tools

    request = request.override(tools=tools)

    print(f"After tools", request.tools)

    return handler(request)
    

#_________________Create Agent_________
agent = create_agent(
    model="google_genai:gemini-3-flash-preview",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)

#_________________Invoke Agent___________
result = agent.invoke(
    {"messages":[{"role": "user", "content": "You have 3 tools: 1.public search, 2.private search, 3.advanced search! can you tell me which tools is available for me from this list since i am authenticated!"}]},
    state={"authenticated": True, "messages":["hi", "hello", "ok"]}
)

#________________Print Result___________
print(result["messages"][-1].content[-1]["text"])