from src.instructions.seo_agent_instruction import get_seo_agent_instructions
from src.middleware.tool_error_handler import handle_tool_errors
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.middleware.log_llm_usage import log_llm_usage
class SEOAgent:
    def __init__(self, llm: ChatOpenAI):
        self.name = "SEO Agent"
        self.description = get_seo_agent_instructions()
        self.model = llm
        self._tools = None
        self._agent = None
        self._tool_warning: Optional[str] = None
        self._mcp_client: Optional[MultiServerMCPClient] = None

    def _build_gsc_server_config(self) -> Dict[str, Any]:
        """Build configuration for the local GSC MCP server."""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        gsc_server_path = project_root / "src" / "tools" / "gsc_server.py"

        python_interpreter = sys.executable

        return {
            "command": python_interpreter,
            "args": [str(gsc_server_path)],
            "transport": "stdio",
            "env": {
                "GSC_CREDENTIALS": os.getenv("GSC_CREDENTIALS", ""),
                "GSC_SKIP_OAUTH": os.getenv("GSC_SKIP_OAUTH", "true"),
            },
        }

    def _build_dataforseo_config(self) -> Optional[Dict[str, Any]]:
        """Build configuration for the optional DataForSEO MCP server."""
        enable_remote = os.getenv("ENABLE_DATAFORSEO_MCP", "true").lower() in ("1", "true", "yes")
        if not enable_remote:
            return None

        default_url = "https://dataforseo-mcp-worker.hitesh-solanki.workers.dev/mcp"
        dataforseo_url = os.getenv("DATAFORSEO_MCP_URL", default_url).strip()
        if not dataforseo_url:
            return None

        config: Dict[str, Any] = {
            "transport": "streamable_http",
            "url": dataforseo_url,
        }

        timeout = os.getenv("DATAFORSEO_MCP_TIMEOUT")
        if timeout:
            try:
                config["timeout"] = float(timeout)
            except ValueError:
                pass

        auth_header = os.getenv("DATAFORSEO_MCP_AUTH_HEADER")
        if auth_header:
            config["headers"] = {"Authorization": auth_header.strip()}

        return config

    def _build_server_config(self, include_dataforseo: bool = True) -> Dict[str, Any]:
        """Build the MCP server configuration dictionary."""
        servers: Dict[str, Any] = {
            "gscServer": self._build_gsc_server_config()
        }

        if include_dataforseo:
            dataforseo_config = self._build_dataforseo_config()
            if dataforseo_config:
                servers["dataforseo"] = dataforseo_config

        return servers

    def get_mcp_client(self, include_dataforseo: bool = True):
        """Initialize and return the MCP client with configured servers."""
        config = self._build_server_config(include_dataforseo=include_dataforseo)
        return MultiServerMCPClient(config)
    
    async def get_tools(self):
        """Get tools from MCP clients. Caches tools for reuse."""
        if self._tools is None:
            self._tool_warning = None

            # Try loading with both servers first
            client = self.get_mcp_client(include_dataforseo=True)
            try:
                self._mcp_client = client
                self._tools = await client.get_tools()
                return self._tools
            except Exception as exc:
                # If DataForSEO is enabled and fails, fall back to GSC-only
                server_config = self._build_server_config(include_dataforseo=True)
                if "dataforseo" in server_config:
                    fallback_client = self.get_mcp_client(include_dataforseo=False)
                    try:
                        details = str(exc)
                        if len(details) > 500:
                            details = details[:500] + "... (truncated)"
                        self._tool_warning = (
                            "⚠️ DataForSEO MCP server could not be reached. "
                            "Continuing with Google Search Console tools only.\n"
                            f"Details: {details}"
                        )
                        self._mcp_client = fallback_client
                        self._tools = await fallback_client.get_tools()
                        return self._tools
                    except Exception as fallback_exc:
                        raise fallback_exc from exc
                # No fallback available, re-raise
                raise
        return self._tools

    def get_tool_warning(self) -> Optional[str]:
        """Return any warning generated while loading tools."""
        return self._tool_warning

    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt and invalidate the cached agent."""
        self.description = new_prompt
        # Invalidate the cached agent so it will be recreated with the new prompt
        self._agent = None

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
                middleware=[handle_tool_errors, log_llm_usage]
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