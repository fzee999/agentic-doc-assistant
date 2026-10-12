"""Command line interface: `docagent ingest | chat | ask | sources`."""

from __future__ import annotations

import argparse
import sys
import uuid


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="docagent", description="Agentic document assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add a folder or file to the knowledge base")
    p.add_argument("path")

    p = sub.add_parser("chat", help="interactive chat with memory")
    p.add_argument("--thread", help="resume a conversation by id")

    p = sub.add_parser("ask", help="ask one question (use --thread to continue a conversation)")
    p.add_argument("question")
    p.add_argument("--thread", default="cli")

    sub.add_parser("sources", help="list indexed documents")

    p = sub.add_parser("eval", help="score routing, tool choice and answers on an eval set (calls the LLM)")
    p.add_argument("--file", default="eval/agent_questions.jsonl")
    p.add_argument("--only", help="comma-separated item ids to run")
    p.add_argument("--out", help="save full results (answers, steps) as JSON")

    args = parser.parse_args(argv)
    if args.command == "eval":
        _eval(args)
        return
    if args.command == "ingest":
        from ragpipe import ingest_path

        print(f"Ingesting {args.path} ...")
        counts = ingest_path(args.path)
        print(f"Indexed {counts['chunks']} chunks from {counts['files']} files.")
    elif args.command == "sources":
        from ragpipe import list_sources

        for row in list_sources():
            print(f"{row['chunks']:>5} chunks  {row['source']:<45} {row['title']}")
    elif args.command == "ask":
        _run_turn(_graph(), args.question, args.thread)
    else:
        _chat(args.thread or uuid.uuid4().hex[:8])


def _graph():
    from docagent.graph import build_graph
    from docagent.memory import sqlite_checkpointer

    return build_graph(checkpointer=sqlite_checkpointer())


def _run_turn(graph, text: str, thread_id: str) -> None:
    from docagent.graph import stream_turn

    for node, step in stream_turn(graph, text, thread_id):
        if node == "answer":
            print(f"\n{step}\n")
        else:
            print(f"  [{node}] {step}")


def _eval(args: argparse.Namespace) -> None:
    import json

    from docagent.evaluate import load_items, run_eval

    items = load_items(args.file)
    if args.only:
        wanted = set(args.only.split(","))
        items = [item for item in items if item["id"] in wanted]
    result = run_eval(items)

    def mark(ok: bool | None) -> str:
        return "-" if ok is None else ("ok" if ok else "FAIL")

    print(f"{'id':<28} {'route':<26} {'tools':<6} {'answer':<6} secs")
    for row in result["rows"]:
        route = f"{row['route']} {mark(row['route_ok'])}"
        print(f"{row['id']:<28} {route:<26} {mark(row['tools_ok']):<6} {mark(row['answer_ok']):<6} {row['seconds']}")
    print(f"\nrouting_accuracy={result['routing_accuracy']}  tool_accuracy={result['tool_accuracy']}  "
          f"answer_accuracy={result['answer_accuracy']}  mean_seconds={result['mean_seconds']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved details to {args.out}")


def _chat(thread_id: str) -> None:
    graph = _graph()
    print(f"Conversation {thread_id} (resume later with --thread {thread_id}). Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            _run_turn(graph, text, thread_id)


if __name__ == "__main__":
    main()
