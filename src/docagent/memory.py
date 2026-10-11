"""Chat memory: LangGraph checkpointers keyed by `thread_id`, so each conversation keeps its own history."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver


def memory_db_path() -> str:
    load_dotenv(find_dotenv(usecwd=True))
    return os.getenv("MEMORY_DB", ".memory/checkpoints.sqlite")


def sqlite_checkpointer(path: str | None = None) -> SqliteSaver:
    """Persistent memory that survives restarts. Use InMemorySaver for tests."""
    path = path or memory_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit serves reruns from different threads.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def list_threads(checkpointer: SqliteSaver, limit: int = 50) -> list[str]:
    """Most recent conversation ids first."""
    threads: list[str] = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in threads:
            threads.append(thread_id)
            if len(threads) == limit:
                break
    return threads
