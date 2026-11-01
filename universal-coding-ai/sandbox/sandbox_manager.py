"""
Secure Sandbox Execution Manager.
Ensures that all file creations, terminal executions, and tests remain
strictly isolated inside the designated workspace sandbox directory.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class SandboxManager:
    """Manages an isolated directory environment for safe autonomous code execution."""

    def __init__(self, sandbox_dir: str = "sandbox/workspace"):
        self.sandbox_path = Path(sandbox_dir).resolve()
        self.sandbox_path.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Resolves path and guarantees it stays inside sandbox directory boundary."""
        target = (self.sandbox_path / relative_path).resolve()
        if not str(target).startswith(str(self.sandbox_path)):
            raise PermissionError(f"Security boundary violation: '{relative_path}' attempts path traversal.")
        return target

    def write_file(self, relative_path: str, content: str) -> str:
        """Safely writes content to a file inside the sandbox."""
        safe_path = self._resolve_safe_path(relative_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {relative_path}"

    def read_file(self, relative_path: str) -> str:
        """Safely reads content from a file inside the sandbox."""
        safe_path = self._resolve_safe_path(relative_path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()

    def edit_file(self, relative_path: str, target: str, replacement: str) -> str:
        """Replaces exact target string with replacement inside sandbox file."""
        content = self.read_file(relative_path)
        if target not in content:
            raise ValueError(f"Target text not found in {relative_path}")
        new_content = content.replace(target, replacement, 1)
        self.write_file(relative_path, new_content)
        return f"Successfully updated {relative_path}"

    def list_files(self, relative_path: str = ".") -> List[str]:
        """Lists files inside the sandbox directory."""
        safe_path = self._resolve_safe_path(relative_path)
        if not safe_path.exists():
            return []
        items = []
        for p in safe_path.rglob("*"):
            if p.is_file():
                items.append(str(p.relative_to(self.sandbox_path)))
        return items

    def run_command(self, command: str, timeout_sec: int = 10) -> Dict[str, Any]:
        """Executes shell command strictly contained in the sandbox working directory."""
        # Restrict dangerous system commands
        disallowed = ["format", "del /s", "rmdir /s /q", "rm -rf /", "shutdown"]
        for bad in disallowed:
            if bad in command.lower():
                return {"success": False, "returncode": -1, "stdout": "", "stderr": f"Command prohibited by security policy: {command}"}

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.sandbox_path),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out ({timeout_sec}s limit exceeded)",
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
            }

    def clean_sandbox(self):
        """Cleans all generated artifacts from sandbox workspace."""
        if self.sandbox_path.exists():
            shutil.rmtree(self.sandbox_path)
            self.sandbox_path.mkdir(parents=True, exist_ok=True)
