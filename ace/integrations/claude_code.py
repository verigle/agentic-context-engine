"""
Claude Code integration for ACE framework.

This module provides ACEClaudeCode, a wrapper for Claude Code CLI
that automatically learns from execution feedback.

Example:
    from ace.integrations import ACEClaudeCode

    agent = ACEClaudeCode(working_dir="./my_project")
    result = agent.run(task="Refactor the auth module")
    agent.save_playbook("learned.json")
"""

import subprocess
import shutil
import json
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from ..llm_providers import LiteLLMClient
from ..playbook import Playbook
from ..roles import Reflector, Curator, GeneratorOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_playbook_context


# Check if claude CLI is available
CLAUDE_CODE_AVAILABLE = shutil.which("claude") is not None


@dataclass
class ClaudeCodeResult:
    """Result from Claude Code execution."""

    success: bool
    output: str
    execution_trace: str
    returncode: int
    error: Optional[str] = None


class ACEClaudeCode:
    """
    Claude Code with ACE learning capabilities.

    Executes tasks via Claude Code CLI and learns from execution.
    Drop-in wrapper that automatically:
    - Injects learned strategies into prompts
    - Reflects on execution results
    - Updates playbook with new learnings

    Usage:
        # Simple usage
        agent = ACEClaudeCode(working_dir="./project")
        result = agent.run(task="Add unit tests for utils.py")

        # Reuse across tasks (learns from each)
        agent = ACEClaudeCode(working_dir="./project")
        agent.run(task="Task 1")
        agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_playbook("expert.json")

        # Start with existing knowledge
        agent = ACEClaudeCode(
            working_dir="./project",
            playbook_path="expert.json"
        )
        agent.run(task="New task")
    """

    def __init__(
        self,
        working_dir: str,
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        ace_max_tokens: int = 2048,
        playbook: Optional[Playbook] = None,
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        timeout: int = 600,
    ):
        """
        Initialize ACEClaudeCode.

        Args:
            working_dir: Directory where Claude Code will execute
            ace_model: Model for ACE learning (Reflector/Curator)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            ace_max_tokens: Max tokens for ACE learning LLM
            playbook: Existing Playbook instance
            playbook_path: Path to load playbook from
            is_learning: Enable/disable ACE learning
            timeout: Execution timeout in seconds (default: 600)
        """
        if not CLAUDE_CODE_AVAILABLE:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: " "https://claude.ai/code"
            )

        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.is_learning = is_learning
        self.timeout = timeout

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        elif playbook:
            self.playbook = playbook
        else:
            self.playbook = Playbook()

        # Create ACE LLM (for Reflector/Curator)
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.ace_llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

    def run(self, task: str, context: str = "") -> ClaudeCodeResult:
        """
        Execute task via Claude Code with ACE learning.

        Args:
            task: Task description for Claude Code
            context: Additional context (optional)

        Returns:
            ClaudeCodeResult with execution details
        """
        # 1. INJECT: Add playbook context if learning enabled and has bullets
        if self.is_learning and self.playbook.bullets():
            playbook_context = wrap_playbook_context(self.playbook)
            prompt = (
                f"{task}\n\n{context}\n\n{playbook_context}"
                if context
                else f"{task}\n\n{playbook_context}"
            )
        else:
            prompt = f"{task}\n\n{context}" if context else task

        # 2. EXECUTE: Run Claude Code
        result = self._execute_claude_code(prompt)

        # 3. LEARN: Run ACE learning if enabled
        if self.is_learning:
            self._learn_from_execution(task, result)

        return result

    def _execute_claude_code(self, prompt: str) -> ClaudeCodeResult:
        """Execute Claude Code CLI with prompt."""
        try:
            # Filter out ANTHROPIC_API_KEY so Claude Code uses subscription auth
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

            result = subprocess.run(
                ["claude", "--print", "--output-format=stream-json", "--verbose"],
                input=prompt,
                text=True,
                cwd=str(self.working_dir),
                capture_output=True,
                timeout=self.timeout,
                env=env,
            )

            execution_trace, summary = self._parse_stream_json(result.stdout)

            return ClaudeCodeResult(
                success=result.returncode == 0,
                output=summary,
                execution_trace=execution_trace,
                returncode=result.returncode,
                error=result.stderr[:500] if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                success=False,
                output="",
                execution_trace="",
                returncode=-1,
                error=f"Execution timed out after {self.timeout}s",
            )
        except Exception as e:
            return ClaudeCodeResult(
                success=False,
                output="",
                execution_trace="",
                returncode=-1,
                error=str(e),
            )

    def _parse_stream_json(self, stdout: str) -> Tuple[str, str]:
        """
        Parse stream-json output from Claude Code.

        Args:
            stdout: Raw stream-json output

        Returns:
            Tuple of (execution_trace, final_summary)
        """
        trace_parts = []
        final_text = ""
        step_num = 0

        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                if event_type == "assistant":
                    message = event.get("message", {})
                    for block in message.get("content", []):
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text.strip():
                                    trace_parts.append(f"[Reasoning] {text[:300]}")
                                    final_text = text
                            elif block.get("type") == "tool_use":
                                step_num += 1
                                tool_name = block.get("name", "unknown")
                                tool_input = block.get("input", {})
                                # Format tool call
                                if tool_name in ["Read", "Glob", "Grep"]:
                                    target = tool_input.get(
                                        "file_path"
                                    ) or tool_input.get("pattern", "")
                                    trace_parts.append(
                                        f"[Step {step_num}] {tool_name}: {target}"
                                    )
                                elif tool_name in ["Write", "Edit"]:
                                    target = tool_input.get("file_path", "")
                                    trace_parts.append(
                                        f"[Step {step_num}] {tool_name}: {target}"
                                    )
                                elif tool_name == "Bash":
                                    cmd = tool_input.get("command", "")[:80]
                                    trace_parts.append(f"[Step {step_num}] Bash: {cmd}")
                                else:
                                    trace_parts.append(f"[Step {step_num}] {tool_name}")
            except json.JSONDecodeError:
                continue

        execution_trace = (
            "\n".join(trace_parts) if trace_parts else "(No trace captured)"
        )

        # Extract summary from final text
        if final_text:
            paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            summary = paragraphs[-1][:300] if paragraphs else final_text[:300]
        else:
            summary = f"Completed {step_num} steps"

        return execution_trace, summary

    def _learn_from_execution(self, task: str, result: ClaudeCodeResult):
        """Run ACE learning pipeline after execution."""
        # Create GeneratorOutput for Reflector
        generator_output = GeneratorOutput(
            reasoning=result.execution_trace,
            final_answer=result.output,
            bullet_ids=[],  # External agents don't pre-select bullets
            raw={
                "success": result.success,
                "returncode": result.returncode,
            },
        )

        # Build feedback
        status = "succeeded" if result.success else "failed"
        feedback = f"Claude Code task {status}"
        if result.error:
            feedback += f"\nError: {result.error}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Run Curator
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"task: {task}",
            progress=f"Claude Code: {task}",
        )

        # Update playbook
        self.playbook.apply_delta(curator_output.delta)

    def save_playbook(self, path: str):
        """Save learned playbook to file."""
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """Load playbook from file."""
        self.playbook = Playbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current playbook strategies as formatted text."""
        if not self.playbook.bullets():
            return ""
        return wrap_playbook_context(self.playbook)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only)."""
        self.is_learning = False


__all__ = ["ACEClaudeCode", "ClaudeCodeResult", "CLAUDE_CODE_AVAILABLE"]
