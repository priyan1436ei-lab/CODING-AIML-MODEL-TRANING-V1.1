"""
Agent Workspace Tools.
Provides structured tool invocation interface for Autonomous Software Engineering.
"""

import json
from typing import Dict, Any, List, Optional
from sandbox.sandbox_manager import SandboxManager


class ToolRegistry:
    """Manages tool execution and provides schema definitions to the LLM agent."""

    def __init__(self, sandbox: Optional[SandboxManager] = None):
        self.sandbox = sandbox or SandboxManager()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns JSON schemas for agent tool selection."""
        return [
            {
                "name": "write_file",
                "description": "Create or overwrite a source code file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path (e.g. main.py)"},
                        "content": {"type": "string", "description": "Full file source code"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file contents from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path to read"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "edit_file",
                "description": "Replace an exact target snippet in a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path"},
                        "target": {"type": "string", "description": "Exact text to find and replace"},
                        "replacement": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "target", "replacement"],
                },
            },
            {
                "name": "run_command",
                "description": "Execute a terminal shell command inside the sandboxed workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command string to run"},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "run_tests",
                "description": "Run Python unit tests using pytest or python unittest runner.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_file": {"type": "string", "description": "Relative path to test file"},
                    },
                    "required": ["test_file"],
                },
            },
            {
                "name": "list_files",
                "description": "List all files present in the sandboxed workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches tool execution to the appropriate sandbox handler."""
        try:
            if tool_name == "write_file":
                msg = self.sandbox.write_file(arguments["path"], arguments["content"])
                return {"success": True, "result": msg}

            elif tool_name == "read_file":
                content = self.sandbox.read_file(arguments["path"])
                return {"success": True, "result": content}

            elif tool_name == "edit_file":
                msg = self.sandbox.edit_file(arguments["path"], arguments["target"], arguments["replacement"])
                return {"success": True, "result": msg}

            elif tool_name == "list_files":
                files = self.sandbox.list_files()
                return {"success": True, "result": files}

            elif tool_name == "run_command":
                res = self.sandbox.run_command(arguments["command"])
                return res

            elif tool_name == "run_tests":
                test_file = arguments["test_file"]
                cmd = f"python -m unittest {test_file}"
                res = self.sandbox.run_command(cmd)
                return res

            else:
                return {"success": False, "error": f"Unknown tool: '{tool_name}'"}

        except Exception as e:
            return {"success": False, "error": str(e)}
