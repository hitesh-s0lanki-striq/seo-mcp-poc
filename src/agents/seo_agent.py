from src.instructions.seo_agent_instruction import seo_agent_instruction
from src.middleware.tool_error_handler import handle_tool_errors
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import sys
from pathlib import Path


class SEOAgent:
    def __init__(self, llm: ChatOpenAI):
        self.name = "SEO Agent"
        self.description = seo_agent_instruction()
        self.model = llm
        self._tools = None
        self._agent = None
        
    def get_mcp_client(self):
        """Initialize and return the MCP client with configured servers."""
        # Get the path to gsc_server.py relative to this file
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        gsc_server_path = project_root / "src" / "tools" / "gsc_server.py"
        
        # Get Python interpreter (use current interpreter)
        python_interpreter = sys.executable
        
        return MultiServerMCPClient(
            {
                "gscServer": {
                    "command": python_interpreter,
                    "args": [str(gsc_server_path)],
                    "transport": "stdio",
                    "env": {
                        "GSC_CREDENTIALS": os.getenv("GSC_CREDENTIALS", ""),
                        "GSC_SKIP_OAUTH": os.getenv("GSC_SKIP_OAUTH", "true")
                    }
                },
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