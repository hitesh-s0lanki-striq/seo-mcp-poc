# pip install openai-agents
import asyncio, os
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings

async def main():
    # If you added an auth header to your server, include it here.
    # Otherwise omit "headers".
    server = MCPServerStreamableHttp(
        name="dataforseo",
        params={
            "url": "http://localhost:3000/mcp",
            # "headers": {"Authorization": "Basic base64(username:password)"},
            "timeout": 15,
        },
        cache_tools_list=True,  # faster subsequent runs
        max_retry_attempts=3,
    )

    async with server:
        agent = Agent(
            name="SEO Agent",
            instructions="Use the DataForSEO MCP tools to answer SEO questions.",
            mcp_servers=[server],
            # Force the model to use tools when needed
            model_settings=ModelSettings(tool_choice="auto"),
        )

        q = (
            "Get the top 3 Google result domains for 'iphone 16 price in india' "
            "from SERP (desktop, Google.in) and summarise briefly."
        )
        result = await Runner.run(agent, q)
        print(result.final_output)

asyncio.run(main())
