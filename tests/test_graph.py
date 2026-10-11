from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from docagent.graph import Route, build_graph, render_transcript, stream_turn
from docagent.prompts import OUT_OF_SCOPE_REPLY


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that replays scripted responses and accepts bind_tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*responses):
    return ScriptedModel(messages=iter(responses))


def fixed_router(route):
    return lambda messages: Route(route=route, reason="test")


def search_call(query, call_id="call_1"):
    return AIMessage("", tool_calls=[{"name": "search_documents", "args": {"query": query}, "id": call_id}])


def config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def test_documents_route_runs_the_tool_loop(vectorstore):
    graph = build_graph(
        agent_model=scripted(search_call("expense deadline"), AIMessage("Within 90 days [policy/expenses.md].")),
        router=fixed_router("documents"),
        chat_model=scripted(),
        checkpointer=InMemorySaver(),
    )
    state = graph.invoke({"messages": [HumanMessage("When are expenses due?")]}, config("t1"))

    assert state["route"] == "documents"
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1 and "policy/expenses.md" in tool_messages[0].text
    assert state["messages"][-1].text == "Within 90 days [policy/expenses.md]."


def test_summaries_are_delivered_without_a_second_agent_pass(vectorstore, monkeypatch):
    import docagent.tools

    monkeypatch.setattr(docagent.tools, "map_reduce_summary", lambda text, source, title, focus="": "- Claims expire after 90 days.")
    summarize_call = AIMessage("", tool_calls=[{"name": "summarize_document", "args": {"source": "expenses"}, "id": "call_s"}])
    graph = build_graph(agent_model=scripted(summarize_call), router=fixed_router("documents"),
                        chat_model=scripted(), checkpointer=InMemorySaver())

    steps = list(stream_turn(graph, "Summarize the expense policy", "t1"))
    assert [node for node, _ in steps] == ["route_question", "agent", "tools", "deliver", "answer"]
    assert steps[-1][1] == "Summary of [policy/expenses.md]:\n- Claims expire after 90 days."


def test_out_of_scope_skips_the_agent(vectorstore):
    graph = build_graph(agent_model=scripted(), router=fixed_router("out_of_scope"),
                        chat_model=scripted(), checkpointer=InMemorySaver())
    state = graph.invoke({"messages": [HumanMessage("Write me a poem")]}, config("t1"))
    assert state["messages"][-1].text == OUT_OF_SCOPE_REPLY


def test_conversational_reply_is_text_only(vectorstore):
    graph = build_graph(agent_model=scripted(), router=fixed_router("conversational"),
                        chat_model=scripted(AIMessage("Hi! Ask me about your documents.")),
                        checkpointer=InMemorySaver())
    state = graph.invoke({"messages": [HumanMessage("hello")]}, config("t1"))
    assert state["messages"][-1].content == "Hi! Ask me about your documents."


def test_memory_is_per_thread_and_history_is_append_only(vectorstore):
    graph = build_graph(
        agent_model=scripted(
            search_call("expenses"), AIMessage("90 days [policy/expenses.md]."),
            search_call("passwords", "call_2"), AIMessage("15 characters [nist/passwords.pdf p.4]."),
        ),
        router=fixed_router("documents"),
        chat_model=scripted(),
        checkpointer=InMemorySaver(),
    )
    first = graph.invoke({"messages": [HumanMessage("Expense deadline?")]}, config("a"))["messages"]
    second = graph.invoke({"messages": [HumanMessage("And password length?")]}, config("a"))["messages"]

    # Turn 2 only appended to turn 1's messages, never edited them (keeps Claude's thinking blocks valid).
    assert [m.id for m in second[:len(first)]] == [m.id for m in first]
    assert [m.content for m in second[:len(first)]] == [m.content for m in first]
    assert sum(isinstance(m, HumanMessage) for m in second) == 2

    other = graph.get_state(config("b")).values
    assert not other.get("messages")


def test_stream_turn_reports_each_step(vectorstore):
    graph = build_graph(
        agent_model=scripted(search_call("expenses"), AIMessage("90 days.")),
        router=fixed_router("documents"),
        chat_model=scripted(),
        checkpointer=InMemorySaver(),
    )
    steps = list(stream_turn(graph, "Expense deadline?", "t1"))
    assert [node for node, _ in steps] == ["route_question", "agent", "tools", "agent", "answer"]
    assert "search_documents" in steps[1][1]
    assert steps[-1] == ("answer", "90 days.")


def test_transcript_skips_tool_traffic():
    messages = [
        HumanMessage("Expense deadline?"),
        search_call("expenses"),
        ToolMessage("passages...", tool_call_id="call_1"),
        AIMessage("90 days."),
    ]
    assert render_transcript(messages) == "User: Expense deadline?\n\nAssistant: 90 days."
