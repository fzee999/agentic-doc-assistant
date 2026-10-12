"""Agent evaluation: does it route correctly, call the right tools, and answer correctly?

Each item in eval/agent_questions.jsonl gives the expected route, optionally the tools that must be
called, and key phrases the answer must contain ("a|b" accepts either). Items with `history` replay
those turns first in the same thread, to test memory. Every item runs in a fresh thread with
in-memory checkpoints, so the app's saved conversations are untouched.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from ragpipe.evaluate import answer_matches

from docagent.graph import build_graph, stream_turn


def load_items(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_item(graph, item: dict) -> dict:
    thread_id = f"eval-{item['id']}-{uuid.uuid4().hex[:6]}"
    for turn in item.get("history", []):
        for _ in stream_turn(graph, turn, thread_id):
            pass

    started = time.perf_counter()
    steps = list(stream_turn(graph, item["question"], thread_id))
    seconds = time.perf_counter() - started

    state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    messages = state["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    tools_called = sorted({
        call["name"]
        for m in messages[last_human:] if isinstance(m, AIMessage)
        for call in m.tool_calls
    })
    answer = steps[-1][1]
    expected_tools = item.get("tools")
    expected_phrases = item.get("answer_contains")
    return {
        "id": item["id"],
        "route": state.get("route"),
        "route_ok": state.get("route") == item["route"],
        "tools": tools_called,
        "tools_ok": set(expected_tools) <= set(tools_called) if expected_tools else None,
        "answer_ok": answer_matches(answer, expected_phrases) if expected_phrases else None,
        "seconds": round(seconds, 1),
        "steps": [f"[{node}] {step}" for node, step in steps[:-1]],
        "answer": answer,
    }


def run_eval(items: list[dict], graph=None) -> dict:
    graph = graph or build_graph(checkpointer=InMemorySaver())
    rows = [run_item(graph, item) for item in items]

    def rate(key: str) -> float | None:
        scored = [row[key] for row in rows if row[key] is not None]
        return round(sum(scored) / len(scored), 2) if scored else None

    return {
        "routing_accuracy": rate("route_ok"),
        "tool_accuracy": rate("tools_ok"),
        "answer_accuracy": rate("answer_ok"),
        "mean_seconds": round(sum(row["seconds"] for row in rows) / len(rows), 1),
        "rows": rows,
    }
