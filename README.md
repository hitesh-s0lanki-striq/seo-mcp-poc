# SEO MCP POC

A proof of concept application for SEO analysis using Model Context Protocol (MCP) servers, integrating DataForSEO and Google Search Console.

## Features

- **Chat Interface**: Simple Streamlit-based chat interface for interacting with the SEO agent
- **Streaming Output**: Real-time streaming of agent responses
- **LLM Thinking**: Display of agent's reasoning and thinking process
- **MCP Integration**: Connects to DataForSEO and Google Search Console via MCP servers

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
DATAFORSEO_USERNAME=your_dataforseo_username
DATAFORSEO_PASSWORD=your_dataforseo_password
GSC_CREDENTIALS_PATH=/path/to/service_account_credentials.json
DATAFORSEO_URL=http://localhost:3000/mcp  # Optional, for HTTP server
MODEL=gpt-4o-mini  # Optional, default model
```

3. Configure GSC server paths in `src/agents/seo_agent.py` or via environment variables:
- `GSC_CREDENTIALS_PATH`: Path to service account credentials (defaults to `/Users/Hemant/Downloads/service_account_credentials.json`)
- Update the GSC server command path in `SEOAgent.get_mcp_client()` if needed

## Running the Application

### Streamlit Chat Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features of the Chat Interface

- **Simple Chat UI**: Clean, intuitive chat interface
- **Real-time Streaming**: Responses stream in real-time as the agent processes your request
- **Tool Call Indicators**: See when the agent is calling tools with visual indicators
- **Error Handling**: Robust tool error handling with user-friendly messages
- **Model Selection**: Choose from different OpenAI models via sidebar
- **Chat History**: Maintains conversation history during the session
- **Clear Chat**: Button to clear chat history

## Usage

1. Start the Streamlit app
2. Enter your SEO question in the chat input
3. Watch the agent's response stream in real-time:
   - See tool calls as they happen (🔧 indicator)
   - Watch the response appear character by character
   - The agent will use appropriate tools (DataForSEO, GSC)
   - Handle errors gracefully
   - Provide SEO insights and analysis
4. Use the sidebar to change models or clear chat history

## Project Structure

```
seo-mcp-poc/
├── app.py                  # Main Streamlit application entry point
├── app.ipynb              # Jupyter notebook for testing
├── src/
│   ├── agents/
│   │   └── seo_agent.py   # SEO Agent class with MCP integration
│   ├── instructions/
│   │   └── seo_agent_instruction.py  # Agent system prompts
│   ├── middleware/
│   │   └── tool_error_handler.py     # Tool error handling middleware
│   ├── ui/
│   │   └── app_ui.py      # Streamlit UI components
│   ├── sample.py          # Example script
│   └── sample.ipynb       # Jupyter notebook examples
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## MCP Servers

### DataForSEO
- Provides keyword research, SERP analysis, and SEO data
- Can use HTTP server (if available) or stdio transport

### Google Search Console (GSC)
- Provides website performance and search analytics
- Uses stdio transport
- Requires service account credentials

## Notes

- The agent initializes on first use and is cached in Streamlit session state
- MCP servers are managed automatically
- Tool errors are handled gracefully with user-friendly messages
- The agent uses LangChain's `create_agent` with error handling middleware
- Model selection changes will reinitialize the agent

## Error Handling

The application includes robust error handling for tool execution:
- Tool errors are caught and converted to user-friendly messages
- The agent can retry operations when appropriate
- Error messages are displayed to help users understand what went wrong









