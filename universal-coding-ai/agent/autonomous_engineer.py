"""
Autonomous Software Engineer Agent Runtime.
Executes the full software engineering lifecycle:
User Requirement -> Analysis -> Planning -> Architecture -> Code Generation
-> File Creation -> Execution -> Error Detection -> Debugging -> Testing -> Repair -> Final App.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from inference.generator import CodeGenerator
from tools.workspace_tools import ToolRegistry
from sandbox.sandbox_manager import SandboxManager


class AutonomousSoftwareEngineer:
    """Autonomous Agent powered by the Universal Coding Model."""

    def __init__(
        self,
        generator: Optional[CodeGenerator] = None,
        sandbox: Optional[SandboxManager] = None,
    ):
        self.generator = generator or CodeGenerator()
        self.sandbox = sandbox or SandboxManager()
        self.tools = ToolRegistry(self.sandbox)

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extracts <|tool_call|>{"name": ..., "arguments": ...}<|tool_result|> from model output."""
        pattern = r"<\|tool_call\|>(.*?)(?:<\|tool_result\|>|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
            try:
                call_obj = json.loads(raw_json)
                return call_obj.get("name"), call_obj.get("arguments", {})
            except Exception:
                pass
        return None

    def solve_requirement(
        self,
        requirement: str,
        max_repair_attempts: int = 3,
    ) -> Dict[str, Any]:
        """Executes the full autonomous engineering cycle for a user requirement."""
        print("=" * 80)
        print("      AUTONOMOUS SOFTWARE ENGINEER: TASK EXECUTION")
        print("=" * 80)
        print(f"Goal Requirement: {requirement}\n")

        execution_log = {
            "requirement": requirement,
            "steps": [],
            "status": "in_progress",
            "files_created": [],
            "test_results": {},
            "final_application": "",
        }

        # Step 1: Requirement Analysis & Architectural Planning
        print("[Step 1/6] Analyzing Requirement & Designing Software Architecture...")
        plan_prompt = (
            f"Analyze the following software engineering requirement and plan the application architecture.\n"
            f"Requirement: {requirement}\n"
            f"List required files, primary classes, data structures, and unit test requirements."
        )
        plan_output = self.generator.generate(plan_prompt, max_new_tokens=200, temperature=0.3)
        execution_log["steps"].append({"phase": "planning", "output": plan_output})

        # Step 2: Source Code Implementation
        print("[Step 2/6] Generating Implementation & Application Code...")
        code_prompt = (
            f"Write the complete, self-contained Python application implementing this requirement:\n"
            f"{requirement}\n"
            f"Provide only verified, clean, robust Python code."
        )
        code_output = self.generator.generate(code_prompt, max_new_tokens=300, temperature=0.2)

        # Extract code from markdown or output
        clean_code = code_output
        if "```python" in clean_code:
            clean_code = clean_code.split("```python")[1].split("```")[0].strip()
        elif "```" in clean_code:
            clean_code = clean_code.split("```")[1].split("```")[0].strip()

        # Step 3: Sandboxed File Creation
        print("[Step 3/6] Creating files in Isolated Sandbox Workspace...")
        app_file = "app.py"
        self.tools.execute_tool("write_file", {"path": app_file, "content": clean_code})
        execution_log["files_created"].append(app_file)

        # Step 4: Unit Test Suite Generation
        print("[Step 4/6] Creating Automated Verification Tests...")
        test_prompt = (
            f"Write a Python unittest test suite in `test_app.py` for testing `app.py`.\n"
            f"Requirement: {requirement}\n"
            f"Code:\n{clean_code[:400]}"
        )
        test_code_raw = self.generator.generate(test_prompt, max_new_tokens=250, temperature=0.2)
        test_code = test_code_raw
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0].strip()
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0].strip()

        # Fallback minimal test if test code is missing unittest structure
        if "unittest" not in test_code and "assert" not in test_code:
            test_code = (
                "import unittest\n"
                "import app\n\n"
                "class TestApp(unittest.TestCase):\n"
                "    def test_module_exists(self):\n"
                "        self.assertTrue(hasattr(app, '__file__'))\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )

        test_file = "test_app.py"
        self.tools.execute_tool("write_file", {"path": test_file, "content": test_code})
        execution_log["files_created"].append(test_file)

        # Step 5: Execution, Error Detection & Self-Repair Loop
        print("[Step 5/6] Executing Tests & Monitoring for Defects...")
        attempt = 0
        all_passed = False

        while attempt < max_repair_attempts and not all_passed:
            attempt += 1
            test_res = self.tools.execute_tool("run_tests", {"test_file": test_file})
            execution_log["test_results"][f"attempt_{attempt}"] = test_res

            if test_res.get("success", False):
                print(f" >>> All tests passed on attempt {attempt}!")
                all_passed = True
                break
            else:
                stderr = test_res.get("stderr") or test_res.get("stdout") or "Test failed"
                print(f" >>> Error detected on attempt {attempt}:\n     {stderr[:120]}...")
                print(f" >>> Performing autonomous code repair...")

                repair_prompt = (
                    f"Fix the bug in `app.py`.\n"
                    f"Error traceback:\n{stderr[:300]}\n"
                    f"Current code:\n{clean_code[:400]}\n"
                    f"Provide corrected Python code:"
                )
                repaired = self.generator.generate(repair_prompt, max_new_tokens=300, temperature=0.2)
                if "```python" in repaired:
                    clean_code = repaired.split("```python")[1].split("```")[0].strip()
                elif "```" in repaired:
                    clean_code = repaired.split("```")[1].split("```")[0].strip()
                else:
                    clean_code = repaired.strip()

                self.tools.execute_tool("write_file", {"path": app_file, "content": clean_code})

        # Step 6: Final Verification & Delivery
        print("[Step 6/6] Finalizing Verified Software Artifacts...")
        execution_log["status"] = "success" if all_passed else "completed_with_warnings"
        execution_log["final_application"] = clean_code
        execution_log["workspace_files"] = self.tools.execute_tool("list_files", {})["result"]

        print("=" * 80)
        print(f"Autonomous Engineer Status: {execution_log['status'].upper()}")
        print(f"Generated Files: {execution_log['workspace_files']}")
        print("=" * 80)
        return execution_log
