"""Agentic document assistant: a LangGraph agent over the rag-knowledge-pipeline retrieval layer."""

from docagent.graph import build_graph, stream_turn

__all__ = ["build_graph", "stream_turn"]
