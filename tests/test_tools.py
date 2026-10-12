from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import docagent.tools
from docagent.tools import (
    list_documents,
    map_reduce_summary,
    resolve_source,
    search_documents,
    summarize_document,
)


def test_long_documents_are_summarized_map_reduce(monkeypatch):
    responses = iter([AIMessage(f"partial {n}") for n in range(1, 4)] + [AIMessage("final summary")])
    fake = GenericFakeChatModel(messages=responses)
    monkeypatch.setattr(docagent.tools, "get_chat_model", lambda **kwargs: fake)
    monkeypatch.setattr(docagent.tools, "_summary_part_chars", lambda: 100)

    text = "\n\n".join(["Paragraph about expenses and reimbursement rules." * 2] * 3)  # 3 parts of ~100 chars
    assert map_reduce_summary(text, "a.md", "A") == "final summary"
    assert next(responses, None) is None  # 3 map calls + 1 reduce call


def test_search_returns_labeled_passages(vectorstore):
    result = search_documents.invoke({"query": "password length"})
    assert result.startswith("[")


def test_source_is_a_soft_filter_that_prefers_the_named_document(vectorstore):
    result = search_documents.invoke({"query": "anything", "source": "nist/passwords.pdf"})
    assert result.startswith("[nist/passwords.pdf p.4]")  # the named document ranks first...
    assert "[policy/expenses.md]" in result               # ...but a wrong guess can't hide other documents


def test_list_documents(vectorstore):
    result = list_documents.invoke({})
    assert "- nist/passwords.pdf: Password Guidance" in result
    assert "- policy/expenses.md: Expense Policy" in result


def test_guessed_document_names_resolve_to_real_paths(vectorstore):
    assert resolve_source("policy/expenses.md") == "policy/expenses.md"   # exact
    assert resolve_source("expenses.md") == "policy/expenses.md"          # bare file name
    assert resolve_source("Expense Policy") == "policy/expenses.md"       # title
    assert resolve_source("nist/password.pdf") == "nist/passwords.pdf"    # near miss
    assert resolve_source("quarterly-roadmap.md") is None                 # nothing close
    assert resolve_source("missing.md") is None                           # file extensions don't count


def test_summarize_unknown_document_points_to_list_documents(vectorstore):
    result = summarize_document.invoke({"source": "missing.md"})
    assert "list_documents" in result
