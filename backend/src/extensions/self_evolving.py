import json
import os

from backend.src.rag.llm_client import llm_client

EVOLUTION_FILE = os.path.join("knowledge", "evolution.json")


def load_state():
    if not os.path.exists(EVOLUTION_FILE):
        return {"rules": [], "patterns": []}
    with open(EVOLUTION_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state):
    os.makedirs(os.path.dirname(EVOLUTION_FILE), exist_ok=True)
    with open(EVOLUTION_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


async def evolve_system(issue_data, fix_data):
    """
    Learn from issues + fixes and generate improved scanning logic.
    """

    current = load_state()

    prompt = f"""
Given the issue data and fix steps below:

ISSUE:
{issue_data}

FIX:
{fix_data}

Generate:
1. A generalized pattern for future detection
2. A rule that can be auto-applied
3. Risk & severity signature
"""

    new_rule = await llm_client.ask(prompt)

    current["rules"].append(new_rule)
    save_state(current)

    return new_rule
