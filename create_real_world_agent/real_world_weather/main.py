from langchain.agents import create_agent, structured_output
from langchain.tools import tool, ToolRuntime

from langgraph.checkpoint.memory import InMemorySaver

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """
    You are an helpful AI assistant who specialized in weather & forecasting related updates and news.

    You have access to two tools:

    - get_weather_for_location: use this to get the weather for specific location
    - get_user_location: user this to get the location of user

    If a user asks you for the weather, make sure you know location.
    Use the get_user_location tool to find their location.
"""

@tool
def get_weather_for_location(location: str) -> str:
    """Get weather for given location or city"""
    return f"It's always sunny in {location}!"


class Context(BaseModel):
    """Custom runtime context schema."""
    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Get user's location based on user ID."""
    user_id = None
    try:
        user_id = runtime.context.user_id
    except:
        pass
    return "Florida" if user_id and user_id == "1" else "New York"


tools = [get_user_location, get_weather_for_location]


class ResponseFormat(BaseModel):
    """Response-schema for agent"""
    punny_response: str
    weather_response: str | None = None


agent = create_agent(
    model="google_genai:gemini-3-flash-preview",
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    checkpointer=checkpointer,
    response_format=structured_output.ToolStrategy(ResponseFormat)
)

config = {"configurable": {"thread_id": "1"}}


while True:

    user_input = input("HUMAN🙉 : ")    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )

    print(result)
    print(f"{'=' * 36}")
    print(f"AI🤖 : {result["messages"][-1].content}")