"""Tool usage tracking module to track tool calls by name."""

from collections import defaultdict
from typing import Dict, Optional, Tuple
import threading
import json
from pathlib import Path


class ToolUsageTracker:
    """
    Thread-safe singleton class to track tool usage counts.
    """
    _instance: Optional['ToolUsageTracker'] = None
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
        
        # Determine the stats file path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        self._stats_file = project_root / "data" / "tool_usage_stats.json"
        
        # Create data directory if it doesn't exist
        self._stats_file.parent.mkdir(exist_ok=True)
        
        self._usage_counts: Dict[str, int] = defaultdict(int)
        self._token_counts: Dict[str, int] = defaultdict(int)  # Total tokens per tool
        self._token_counts_list: Dict[str, list] = defaultdict(list)  # Individual token counts per call
        self._lock = threading.Lock()
        self._initialized = True
        
        # Load existing stats from file
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics from the JSON file."""
        if not self._stats_file.exists():
            return
        
        try:
            with open(self._stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            with self._lock:
                # Load usage counts
                if 'usage_counts' in data:
                    self._usage_counts = defaultdict(int, data['usage_counts'])
                
                # Load token counts
                if 'token_counts' in data:
                    self._token_counts = defaultdict(int, data['token_counts'])
                
                # Load token counts list (convert lists back from JSON)
                if 'token_counts_list' in data:
                    self._token_counts_list = defaultdict(list, {
                        k: v if isinstance(v, list) else [] 
                        for k, v in data['token_counts_list'].items()
                    })
        except (json.JSONDecodeError, IOError, OSError) as e:
            # If file is corrupted or can't be read, start fresh
            print(f"Warning: Could not load tool usage stats from {self._stats_file}: {e}")
            self._usage_counts = defaultdict(int)
            self._token_counts = defaultdict(int)
            self._token_counts_list = defaultdict(list)
    
    def _save_stats(self) -> None:
        """Save statistics to the JSON file."""
        try:
            # Prepare data for serialization
            with self._lock:
                data = {
                    'usage_counts': dict(self._usage_counts),
                    'token_counts': dict(self._token_counts),
                    'token_counts_list': dict(self._token_counts_list)
                }
            
            # Write to temporary file first, then rename (atomic write)
            temp_file = self._stats_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self._stats_file)
        except (IOError, OSError) as e:
            # Log error but don't crash the application
            print(f"Warning: Could not save tool usage stats to {self._stats_file}: {e}")

    def track_tool_call(self, tool_name: str, token_count: int = 0) -> None:
        """
        Increment the usage count for a tool and track token usage.
        
        Args:
            tool_name: Name of the tool that was called
            token_count: Number of tokens in the tool output (default: 0)
        """
        if not tool_name:
            return
        
        with self._lock:
            self._usage_counts[tool_name] += 1
            if token_count > 0:
                self._token_counts[tool_name] += token_count
                self._token_counts_list[tool_name].append(token_count)
        
        # Save stats to file after each update
        self._save_stats()

    def get_usage_count(self, tool_name: str) -> int:
        """
        Get the usage count for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Usage count for the tool (0 if never used)
        """
        with self._lock:
            return self._usage_counts.get(tool_name, 0)

    def get_all_usage_stats(self) -> Dict[str, int]:
        """
        Get all tool usage statistics.
        
        Returns:
            Dictionary mapping tool names to their usage counts
        """
        with self._lock:
            return dict(self._usage_counts)

    def get_total_calls(self) -> int:
        """
        Get the total number of tool calls made.
        
        Returns:
            Total number of tool calls
        """
        with self._lock:
            return sum(self._usage_counts.values())

    def get_token_count(self, tool_name: str) -> int:
        """
        Get the total token count for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Total token count for the tool (0 if never used or no tokens tracked)
        """
        with self._lock:
            return self._token_counts.get(tool_name, 0)
    
    def get_token_stats(self, tool_name: str) -> Tuple[int, float, int, int]:
        """
        Get detailed token statistics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tuple of (total_tokens, avg_tokens, min_tokens, max_tokens)
        """
        with self._lock:
            token_list = self._token_counts_list.get(tool_name, [])
            if not token_list:
                return (0, 0.0, 0, 0)
            
            total = sum(token_list)
            avg = total / len(token_list) if token_list else 0.0
            min_tokens = min(token_list)
            max_tokens = max(token_list)
            return (total, avg, min_tokens, max_tokens)
    
    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens across all tools.
        
        Returns:
            Total number of tokens
        """
        with self._lock:
            return sum(self._token_counts.values())
    
    def get_all_token_stats(self) -> Dict[str, int]:
        """
        Get all tool token statistics.
        
        Returns:
            Dictionary mapping tool names to their total token counts
        """
        with self._lock:
            return dict(self._token_counts)
    
    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        with self._lock:
            self._usage_counts.clear()
            self._token_counts.clear()
            self._token_counts_list.clear()
        
        # Save empty stats to file
        self._save_stats()

    def get_sorted_stats(self, reverse: bool = True) -> Dict[str, int]:
        """
        Get usage statistics sorted by count.
        
        Args:
            reverse: If True, sort descending (most used first)
            
        Returns:
            Dictionary of tool names to counts, sorted by count
        """
        with self._lock:
            sorted_items = sorted(
                self._usage_counts.items(),
                key=lambda x: x[1],
                reverse=reverse
            )
            return dict(sorted_items)
    
    def get_sorted_token_stats(self, reverse: bool = True) -> Dict[str, int]:
        """
        Get token statistics sorted by total tokens.
        
        Args:
            reverse: If True, sort descending (most tokens first)
            
        Returns:
            Dictionary of tool names to total token counts, sorted by tokens
        """
        with self._lock:
            sorted_items = sorted(
                self._token_counts.items(),
                key=lambda x: x[1],
                reverse=reverse
            )
            return dict(sorted_items)


# Global instance getter
def get_tracker() -> ToolUsageTracker:
    """Get the global tool usage tracker instance."""
    return ToolUsageTracker()

