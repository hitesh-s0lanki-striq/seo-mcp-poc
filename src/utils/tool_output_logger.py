"""Tool output logging module to log tool outputs to JSON or markdown files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple
import threading


class ToolOutputLogger:
    """
    Thread-safe singleton class to log tool outputs to JSON or markdown files.
    - If output is JSON (dict/list), saves as {tool_name}.json
    - If output is a string that can be parsed as JSON, saves as {tool_name}.json
    - Otherwise, saves as {tool_name}.md
    """
    _instance: Optional['ToolOutputLogger'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Determine the logs directory path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self._logs_dir = project_root / "logs"
        
        # Create logs directory if it doesn't exist
        self._logs_dir.mkdir(exist_ok=True)
        
        self._lock = threading.Lock()
        self._initialized = True
    
    def _sanitize_filename(self, tool_name: str) -> str:
        """
        Sanitize tool name to create a valid filename.
        
        Args:
            tool_name: The tool name
            
        Returns:
            Sanitized filename-safe string
        """
        # Replace invalid filename characters with underscores
        invalid_chars = '<>:"/\\|?*'
        sanitized = tool_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unknown_tool"
        return sanitized
    
    def _extract_json_data(self, result: Any) -> Tuple[Optional[Any], bool]:
        """
        Try to extract JSON data from the result.
        
        Args:
            result: The tool output result
            
        Returns:
            Tuple of (json_data, is_json) where is_json indicates if we found valid JSON
        """
        # If it's already a dict or list, it's JSON
        if isinstance(result, (dict, list)):
            return result, True
        
        # If it's a string, try to parse it as JSON
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed, True
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Check if result has a 'content' attribute that might be JSON
        if hasattr(result, 'content'):
            content = result.content
            # If content is a dict/list, it's JSON
            if isinstance(content, (dict, list)):
                return content, True
            # If content is a string, try to parse it
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    return parsed, True
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # Not JSON
        return None, False
    
    def _format_json_output(self, tool_name: str, json_data: Any, timestamp: datetime, token_count: Optional[int] = None, input_data: Optional[dict] = None) -> str:
        """
        Format the tool output as JSON with metadata.
        
        Args:
            tool_name: Name of the tool
            json_data: The JSON data to save
            timestamp: Timestamp of the execution
            token_count: Optional token count for the output
            input_data: Optional input/arguments passed to the tool
            
        Returns:
            Formatted JSON string
        """
        # Create a wrapper object with metadata
        output = {
            "tool_name": tool_name,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "data": json_data
        }
        
        # Add input if provided
        if input_data is not None:
            output["input"] = input_data
        
        # Add token count if provided
        if token_count is not None:
            output["token_count"] = token_count
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def _format_markdown_output(self, tool_name: str, result: Any, timestamp: datetime) -> str:
        """
        Format the tool output as markdown.
        
        Args:
            tool_name: Name of the tool
            result: The tool output result
            timestamp: Timestamp of the execution
            
        Returns:
            Formatted markdown string
        """
        # Format timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Serialize result
        if isinstance(result, (dict, list)):
            try:
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                result_str = str(result)
        else:
            result_str = str(result)
        
        # Create markdown content
        markdown = f"""## Tool Execution: {tool_name}

**Timestamp:** {timestamp_str}

### Output

```
{result_str}
```

---

"""
        return markdown
    
    def log_tool_output(self, tool_name: str, result: Any, token_count: Optional[int] = None, input_data: Optional[dict] = None) -> None:
        """
        Log tool output to a JSON or markdown file.
        - If output is JSON, saves as {tool_name}.json
        - Otherwise, saves as {tool_name}.md
        
        Args:
            tool_name: Name of the tool
            result: The tool output result
            token_count: Optional token count for the output
            input_data: Optional input/arguments passed to the tool
        """
        if not tool_name:
            tool_name = "unknown_tool"
        
        # Sanitize tool name for filename
        sanitized_name = self._sanitize_filename(tool_name)
        
        try:
            # Get current timestamp
            timestamp = datetime.now()
            
            # Try to extract JSON data
            json_data, is_json = self._extract_json_data(result)
            
            if is_json and json_data is not None:
                # Save as JSON file
                log_file = self._logs_dir / f"{sanitized_name}.json"
                
                # Format as JSON with metadata (including token count and input)
                json_content = self._format_json_output(tool_name, json_data, timestamp, token_count, input_data)
                
                # Append to JSON file (thread-safe)
                with self._lock:
                    # Append mode - creates file if it doesn't exist
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(json_content)
                        f.write('\n')  # Add newline separator between entries
            else:
                # Save as markdown file
                log_file = self._logs_dir / f"{sanitized_name}.md"
                
                # Format as markdown
                markdown_content = self._format_markdown_output(tool_name, result, timestamp)
                
                # Append to markdown file (thread-safe)
                with self._lock:
                    # Append mode - creates file if it doesn't exist
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(markdown_content)
        
        except Exception as e:
            # Log error but don't crash the application
            print(f"Warning: Could not log tool output for {tool_name}: {e}")


# Global instance getter
def get_logger() -> ToolOutputLogger:
    """Get the global tool output logger instance."""
    return ToolOutputLogger()

