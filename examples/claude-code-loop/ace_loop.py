#!/usr/bin/env python3
"""
ACE + Claude Code: Continuous Learning Loop

This script demonstrates using ACEClaudeCode to run Claude Code in a loop,
learning from each task execution. Tasks are read from a TODO.md file.

Usage:
    python ace_loop.py                    # Interactive mode
    AUTO_MODE=true python ace_loop.py     # Fully automatic
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from ace.integrations import ACEClaudeCode

load_dotenv()

# Configuration
AUTO_MODE = os.getenv("AUTO_MODE", "false").lower() == "true"
ACE_MODEL = os.getenv("ACE_MODEL", "claude-sonnet-4-5-20250929")
WORKSPACE_DIR = Path(__file__).parent / "workspace"
PLAYBOOK_PATH = Path(__file__).parent / "playbooks" / "ace_typescript.json"


def parse_next_task_from_todo(workspace_dir: Path) -> str | None:
    """
    Parse TODO.md to find next unchecked task.

    Returns:
        Next unchecked task description, or None if all complete
    """
    todo_paths = [workspace_dir / ".agent" / "TODO.md", workspace_dir / "TODO.md"]

    todo_path = None
    for path in todo_paths:
        if path.exists():
            todo_path = path
            print(f"   Found TODO.md at: {path}")
            break

    if not todo_path:
        print(f"   No TODO.md found")
        return None

    content = todo_path.read_text()

    # Look for unchecked tasks: [ ] or - [ ]
    pattern = r"^[\s\-]*\[ \]\s+(.+)$"

    # Skip vague category tasks
    category_indicators = [
        "phase",
        "step",
        "stage",
        "setup",
        "initialization",
        "eslint",
        "linting",
        "ci/cd",
        "configuration",
    ]

    for line in content.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            task = match.group(1).strip()
            task_lower = task.lower()

            # Skip category-level tasks
            if any(ind in task_lower for ind in category_indicators):
                continue

            # Prefer specific tasks
            if len(task.split()) > 2:
                print(f"   Found task: {task[:60]}...")
                return task

    return None


def main():
    """Main orchestration function with continuous loop."""
    print("\n ACE + Claude Code")
    print("=" * 70)

    print(f"\n Initializing (model: {ACE_MODEL})...")
    print(f"   Mode: {'AUTOMATIC' if AUTO_MODE else 'INTERACTIVE'}")

    # Initialize ACEClaudeCode
    PLAYBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    agent = ACEClaudeCode(
        working_dir=str(WORKSPACE_DIR),
        ace_model=ACE_MODEL,
        playbook_path=str(PLAYBOOK_PATH) if PLAYBOOK_PATH.exists() else None,
    )

    print(f" Playbook: {len(list(agent.playbook.bullets()))} strategies")
    print(f" Workspace: {WORKSPACE_DIR}")

    # Read project spec for context
    spec_file = WORKSPACE_DIR / "specs" / "project.md"
    context = ""
    if spec_file.exists():
        context = f"Project specification:\n{spec_file.read_text()[:1000]}..."

    # Initial confirmation
    print("\n" + "=" * 70)
    if not AUTO_MODE:
        response = input(" Start learning loop? (y/n): ")
        if response.lower() != "y":
            print(" Cancelled")
            return

    task_count = 0
    results = []

    # CONTINUOUS LOOP
    while True:
        task_count += 1

        # Determine next task
        if task_count == 1:
            task = """Create .agent/TODO.md with translation tasks.
Use markdown checkbox format: - [ ] for each task.
List specific Python files from source/ that need translation."""
            print(f"\n Task {task_count} (bootstrap): Create TODO.md")
        else:
            task = parse_next_task_from_todo(WORKSPACE_DIR)
            if not task:
                print(f"\n All tasks complete!")
                break
            print(f"\n Task {task_count}: {task[:60]}...")

        # Interactive confirmation
        if not AUTO_MODE and task_count > 1:
            response = input(" Process this task? (y/n/q): ").strip().lower()
            if response == "q":
                break
            elif response != "y":
                continue

        # Execute task
        print(f"\n{'=' * 70}")
        print(f" EXECUTING TASK {task_count}")
        print("=" * 70 + "\n")

        result = agent.run(task=task, context=context)
        results.append(result)

        # Summary
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n Task {task_count}: {status}")
        print(f" Playbook: {len(list(agent.playbook.bullets()))} strategies")

        # Save after each task
        agent.save_playbook(str(PLAYBOOK_PATH))

        if not AUTO_MODE:
            input("\nPress Enter to continue...")

    # Final summary
    print("\n" + "=" * 70)
    print(" COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r.success)
    print(f"\nTasks: {successful}/{len(results)} successful")
    print(f"Playbook: {len(list(agent.playbook.bullets()))} strategies")

    # Show top strategies
    bullets = list(agent.playbook.bullets())
    if bullets:
        print(f"\n Top Learned Strategies:")
        print("-" * 70)
        sorted_bullets = sorted(
            bullets, key=lambda b: b.helpful - b.harmful, reverse=True
        )
        for i, bullet in enumerate(sorted_bullets[:5], 1):
            print(f"{i}. [{bullet.id}] {bullet.content[:70]}...")

    print(f"\n Playbook saved to: {PLAYBOOK_PATH}")


if __name__ == "__main__":
    main()
