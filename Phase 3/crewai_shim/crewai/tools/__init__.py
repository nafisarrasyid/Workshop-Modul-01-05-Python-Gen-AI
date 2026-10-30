"""CrewAI tools compatibility module."""
from __future__ import annotations
from typing import Any, Callable, Optional, Type
from pydantic import BaseModel, Field

class BaseTool:
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def __init__(self, **data: Any):
        for k, v in data.items():
            setattr(self, k, v)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement _run")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

def tool(name: Optional[str] = None, description: Optional[str] = None):
    def decorator(fn: Callable) -> BaseTool:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or "No description provided."
        
        class FuncTool(BaseTool):
            name = tool_name
            description = tool_desc
            def _run(self, *args, **kwargs):
                return fn(*args, **kwargs)
                
        return FuncTool()
    return decorator

__all__ = ["BaseTool", "tool"]
