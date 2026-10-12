r"""The agent graph.

    START -> route_question --documents------> agent <--> tools   (tool loop) -> END
                                                            \--> deliver -----> END   (summaries)
                            --compare--------> plan -> research -> compare_answer -> END
                            --conversational-> conversational ----------------> END
                            --out_of_scope---> out_of_scope ------------------> END

- route_question: the LLM classifies the latest message with structured output.
- agent: the LLM with tools (search, summarize, list) decides what to call and when to stop,
  then answers with citations.
- deliver: a summarize_document result is returned to the user as is (no paraphrasing pass).
- plan / research / compare_answer: for questions that compare documents, the LLM plans one
  search per document (structured output), the searches run deterministically, and the LLM
  answers from the per-document passages. Small models can't be trusted to plan this inside
  the tool loop: left alone, they search once, in the wrong document.
- Memory: a checkpointer stores each thread's messages, so follow-up questions have context.

Design rule: the shared `messages` history is append-only, and the agent always runs with
the same system prompt and tool list. With Claude (RAG_PROVIDER=anthropic), a thinking block
is tied to the exact conversation that produced it, so editing history, changing the system
prompt mid-conversation, or replaying blocks under another prompt invalidates them. That's
why the router, planner, comparison and conversational steps work from a separate plain-text
transcript, and only add text (never thinking blocks) to the history. Local models don't need
this, but it costs nothing and keeps both providers working.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from ragpipe import list_sources, search
from ragpipe.llm import get_chat_model, is_refusal

from docagent.prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPARE_PROMPT,
    CONVERSATIONAL_PROMPT,
    OUT_OF_SCOPE_REPLY,
    PLAN_PROMPT,
    ROUTER_PROMPT,
)
from docagent.tools import TOOLS, _label, _store, resolve_source

MAX_PLANNED_SEARCHES = 4
PASSAGES_PER_SEARCH = 4


class Route(BaseModel):
    route: Literal["documents", "compare", "conversational", "out_of_scope"]
    reason: str = Field(description="One short sentence explaining the choice")


class PlannedSearch(BaseModel):
    document: str = Field(description="Path of the document to search, from the list")
    query: str = Field(description="Short keyword query for what to look up in that document")


class Plan(BaseModel):
    searches: list[PlannedSearch] = Field(description="One search per document involved, 2 to 4 in total")


class AgentState(MessagesState):
    route: str
    route_reason: str
    plan: list[dict]
    research: str


Router = Callable[[list[BaseMessage]], Route]
Planner = Callable[[list[BaseMessage]], Plan]


def render_transcript(messages: list[BaseMessage], limit: int = 12) -> str:
    """Plain-text view of recent user/assistant turns (no tool traffic, no thinking blocks)."""
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.text}")
        elif isinstance(message, AIMessage) and message.text.strip():
            lines.append(f"Assistant: {message.text}")
    return "\n\n".join(lines[-limit:])


def llm_router() -> Router:
    classifier = get_chat_model(effort="low").with_structured_output(Route, method="json_schema")

    def route(messages: list[BaseMessage]) -> Route:
        catalog = "\n".join(f"- {row['title']}" for row in list_sources(_store())) or "- (no documents indexed yet)"
        return classifier.invoke(ROUTER_PROMPT.format(catalog=catalog, transcript=render_transcript(messages)))

    return route


def llm_planner() -> Planner:
    planner = get_chat_model(effort="low").with_structured_output(Plan, method="json_schema")

    def plan(messages: list[BaseMessage]) -> Plan:
        catalog = "\n".join(f"- {row['source']}: {row['title']}" for row in list_sources(_store()))
        return planner.invoke(PLAN_PROMPT.format(catalog=catalog, transcript=render_transcript(messages)))

    return plan


def build_graph(
    *,
    agent_model: BaseChatModel | None = None,
    router: Router | None = None,
    planner: Planner | None = None,
    chat_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the agent. Every dependency is injectable so tests can run without an LLM."""
    answer_llm = agent_model or get_chat_model()
    agent_llm = answer_llm.bind_tools(TOOLS)
    router = router or llm_router()
    planner = planner or llm_planner()
    chat_llm = chat_model or get_chat_model(effort="low")

    def route_question(state: AgentState) -> dict:
        decision = router(state["messages"])
        return {"route": decision.route, "route_reason": decision.reason}

    def agent(state: AgentState) -> dict:
        response = agent_llm.invoke([SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]])
        if is_refusal(response):
            response = AIMessage("Sorry, I can't help with that request.")
        return {"messages": [response]}

    def conversational(state: AgentState) -> dict:
        reply = chat_llm.invoke([
            SystemMessage(CONVERSATIONAL_PROMPT),
            HumanMessage(f"<conversation>\n{render_transcript(state['messages'])}\n</conversation>"),
        ])
        return {"messages": [AIMessage(reply.text)]}  # text only, see module docstring

    def out_of_scope(state: AgentState) -> dict:
        return {"messages": [AIMessage(OUT_OF_SCOPE_REPLY)]}

    def after_tools(state: AgentState) -> str:
        results = _trailing_tool_messages(state["messages"])
        return "deliver" if results and all(m.name in DIRECT_TOOLS for m in results) else "agent"

    def deliver(state: AgentState) -> dict:
        # The tool's output is already the answer (an LLM-written, document-grounded summary), so show
        # it as is instead of asking the model to paraphrase it. Small models tend to garble that step.
        results = _trailing_tool_messages(state["messages"])
        return {"messages": [AIMessage("\n\n".join(m.text for m in results))]}

    def plan(state: AgentState) -> dict:
        searches = planner(state["messages"]).searches[:MAX_PLANNED_SEARCHES]
        return {"plan": [{"document": s.document, "query": s.query} for s in searches]}

    def research(state: AgentState) -> dict:
        sections = []
        for n, step in enumerate(state["plan"], start=1):
            source = resolve_source(step["document"])  # None -> search everything
            hits = search(step["query"], k=PASSAGES_PER_SEARCH, vectorstore=_store(), source=source)
            passages = "\n\n".join(f"[{_label(doc)}]\n{doc.page_content}" for doc, _ in hits) or "(no passages found)"
            sections.append(f"### Search {n}: '{step['query']}' in {source or 'all documents'}\n\n{passages}")
        return {"research": "\n\n".join(sections)}

    def compare_answer(state: AgentState) -> dict:
        reply = answer_llm.invoke([
            SystemMessage(COMPARE_PROMPT),
            HumanMessage(f"<passages>\n{state['research']}\n</passages>\n\n"
                         f"<conversation>\n{render_transcript(state['messages'])}\n</conversation>\n\n"
                         "Answer the user's latest message."),
        ])
        text = "Sorry, I can't help with that request." if is_refusal(reply) else reply.text
        return {"messages": [AIMessage(text)]}  # text only, see module docstring

    graph = StateGraph(AgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("deliver", deliver)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("compare_answer", compare_answer)
    graph.add_node("conversational", conversational)
    graph.add_node("out_of_scope", out_of_scope)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        lambda state: state["route"],
        {"documents": "agent", "compare": "plan", "conversational": "conversational", "out_of_scope": "out_of_scope"},
    )
    graph.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools", otherwise END
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "deliver": "deliver"})
    graph.add_edge("deliver", END)
    graph.add_edge("plan", "research")
    graph.add_edge("research", "compare_answer")
    graph.add_edge("compare_answer", END)
    graph.add_edge("conversational", END)
    graph.add_edge("out_of_scope", END)
    return graph.compile(checkpointer=checkpointer)


# Tools whose output goes straight to the user (no extra LLM pass).
DIRECT_TOOLS = {"summarize_document"}


def _trailing_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    results: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        results.insert(0, message)
    return results


def stream_turn(graph, text: str, thread_id: str) -> Iterator[tuple[str, str]]:
    """Run one user turn, yielding (node, human-readable step) as each node finishes.

    The last item is ("answer", final answer text).
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    for update in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
        for node, values in update.items():
            yield node, describe_step(node, values)
    final = graph.get_state(config).values["messages"][-1]
    yield "answer", final.text


def describe_step(node: str, values: dict) -> str:
    if node == "route_question":
        return f"Routed as '{values['route']}': {values['route_reason']}"
    if node == "plan":
        return "Planned " + "; ".join(f"'{s['query']}' in {s['document']}" for s in values["plan"])
    if node == "research":
        return f"Retrieved passages for {values['research'].count('### Search')} searches"
    messages = values.get("messages", [])
    if node == "agent" and messages and messages[-1].tool_calls:
        calls = ", ".join(f"{call['name']}({_short_args(call['args'])})" for call in messages[-1].tool_calls)
        return f"Calling {calls}"
    if node == "tools":
        return ", ".join(f"{m.name} returned {len(m.text):,} chars" for m in messages if isinstance(m, ToolMessage))
    if node == "deliver":
        return "Returning the summary as the answer"
    if node == "compare_answer":
        return "Writing the comparison"
    return "Writing the answer"


def _short_args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())
