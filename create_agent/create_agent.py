from langchain.agents import create_agent

from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    system_prompt="You are an helpful coding assistant"
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "list out other framework like django"}]}
)

print(result["messages"][-1].content)