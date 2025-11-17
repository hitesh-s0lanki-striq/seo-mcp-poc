from src.instructions.seo_agent_instruction import seo_agent_instruction
from src.middleware.tool_error_handler import handle_tool_errors
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import os


class SEOAgent:
    def __init__(self, llm: ChatOpenAI):
        self.name = "SEO Agent"
        self.description = seo_agent_instruction()
        self.model = llm
        self._tools = None
        self._agent = None
        
    def get_mcp_client(self):
        """Initialize and return the MCP client with configured servers."""
        return MultiServerMCPClient(
            {
                # "gscServer": {
                #     "command": "/Users/Hemant/Desktop/strique/mcp-gsc/.venv/bin/python",
                #     "args": ["/Users/Hemant/Desktop/strique/mcp-gsc/gsc_server.py"],
                #     "transport": "stdio",
                #     "env": {
                #         "GSC_CREDENTIALS_PATH": os.getenv(
                #             "GSC_CREDENTIALS_PATH",
                #             "/Users/Hemant/Downloads/service_account_credentials.json"
                #         ),
                #         "GSC_SKIP_OAUTH": "true"
                #     }
                # },
                "dataforseo": {
                    "transport": "streamable_http",
                    "url": "https://dataforseo-mcp-worker.hitesh-solanki.workers.dev/mcp"
                }
            }
        )
    
    async def get_tools(self):
        """Get tools from MCP clients. Caches tools for reuse."""
        if self._tools is None:
            # Initialize the MCP client
            client = self.get_mcp_client()
            
            # Get the tools
            self._tools = await client.get_tools()
        return self._tools

    async def get_agent(self):
        """Get or create the agent with tools and error handling middleware."""
        if self._agent is None:
            # Get the tools
            tools = await self.get_tools()
            
            # Create the agent with error handling middleware
            self._agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=self.description,
                middleware=[handle_tool_errors]
            )
        
        return self._agent
        
    async def run(self, messages):
        """Run the agent with the given messages."""
        # Get the agent
        agent = await self.get_agent()
        
        # Run the agent
        result = await agent.ainvoke({
            "messages": messages
        })
        
        return result
    
    async def stream(self, messages):
        """Stream agent responses as they are generated."""
        # Get the agent
        agent = await self.get_agent()
        
        # Stream the agent's response
        async for chunk in agent.astream(
            {"messages": messages},
            stream_mode="values"
        ):
            yield chunk