"""Tools the agent can call. Retrieval is delegated to the rag-knowledge-pipeline package (`ragpipe`)."""

from __future__ import annotations

import math
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragpipe import get_settings, get_source_chunks, get_vectorstore, list_sources, search
from ragpipe.llm import get_chat_model
from ragpipe.retrieval import rrf_fuse

from docagent.prompts import SUMMARY_PROMPT


def _store() -> Chroma:
    # Indirection so tests can point the tools at a temporary index.
    return get_vectorstore()


def resolve_source(name: str | None, min_score: float = 0.5) -> str | None:
    """Map a model-supplied document name to a real indexed path.

    Small models often guess paths ("travel-and-expense-policy.md", "CSF 2.0") instead of copying
    them from list_documents. Score each document by the share of the name's words found in its
    path or title, weighting rare words (travel, csf) above common ones (gitlab, md), like IDF.
    Return None if no document scores at least `min_score`.
    """
    if not name:
        return None
    rows = list_sources(_store())
    if any(row["source"] == name for row in rows):
        return name
    doc_tokens = {row["source"]: _tokens(row["source"]) | _tokens(row["title"]) for row in rows}
    n_docs = len(doc_tokens)

    def weight(token: str) -> float:
        df = sum(token in tokens for tokens in doc_tokens.values())
        return math.log(1 + n_docs / df) if df else math.log(1 + n_docs)  # unknown words count as rare

    query = _tokens(name)
    total = sum(weight(t) for t in query)
    if not query or not total:
        return None
    scores = {src: sum(weight(t) for t in query & tokens) / total for src, tokens in doc_tokens.items()}
    ranked = sorted(scores.values(), reverse=True)
    if not ranked or ranked[0] < min_score or (len(ranked) > 1 and ranked[0] == ranked[1]):
        return None  # nothing close, or ambiguous ("gitlab handbook" fits every handbook page)
    return max(scores, key=scores.get)


_FILE_EXTENSIONS = {"md", "markdown", "txt", "pdf", "html", "htm"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _FILE_EXTENSIONS


def _label(doc: Document) -> str:
    meta = doc.metadata
    return meta["source"] + (f" p.{meta['page']}" if "page" in meta else "")


@tool(parse_docstring=True)
def search_documents(query: str, source: str | None = None) -> str:
    """Semantic search over the knowledge base. Returns the most relevant passages, each labeled with its source.

    Args:
        query: What to look for. Use specific keywords from the user's question.
        source: Optional document path, exactly as listed by list_documents, to search inside one document only.
    """
    resolved = resolve_source(source)
    if source and not resolved:
        return f"No document matches '{source}'. Call list_documents for valid paths, or search without `source`."
    k = get_settings().top_k
    hits = search(query, k=k, vectorstore=_store(), source=resolved)
    if resolved:
        # Soft filter: small models often pick the wrong document, so also search everywhere and
        # fuse. Passages from the named document still rank first when they match.
        hits = rrf_fuse(hits, search(query, k=k, vectorstore=_store()), limit=k)
    if not hits:
        return "No passages found. The index may be empty."
    return "\n\n".join(f"[{_label(doc)}]\n{doc.page_content}" for doc, _ in hits)


@tool(parse_docstring=True)
def summarize_document(source: str, focus: str = "") -> str:
    """Summarize one whole document. Use for overview questions like "what does the expense policy cover?".

    Args:
        source: Document path, exactly as listed by list_documents.
        focus: Optional aspect to emphasize, e.g. "reimbursement limits".
    """
    resolved = resolve_source(source)
    if not resolved:
        return f"No document named '{source}'. Call list_documents to see valid paths."
    source = resolved
    chunks = get_source_chunks(source, vectorstore=_store())
    title = chunks[0].metadata.get("title", "")
    # Chunks start with a "<title>\n\n" header (added at indexing time); drop it when stitching the text back together.
    text = "\n\n".join(chunk.page_content.removeprefix(f"{title}\n\n") for chunk in chunks)
    return f"Summary of [{source}]:\n{map_reduce_summary(text, source, title, focus)}"


@tool(parse_docstring=True)
def list_documents() -> str:
    """List every document in the knowledge base with its path and title."""
    rows = list_sources(_store())
    if not rows:
        return "The knowledge base is empty."
    return "\n".join(f"- {row['source']}: {row['title']}" for row in rows)


TOOLS = [search_documents, summarize_document, list_documents]


def map_reduce_summary(text: str, source: str, title: str, focus: str = "", max_rounds: int = 3) -> str:
    """Summarize text of any length.

    If the text doesn't fit in one call, summarize each part (map), then summarize the
    joined partial summaries (reduce), repeating until a single part remains.
    """
    llm = get_chat_model(effort="low")
    part_chars = _summary_part_chars()

    def summarize(part: str, label: str) -> str:
        return llm.invoke([
            SystemMessage(SUMMARY_PROMPT),
            HumanMessage(f'<document source="{source}" title="{title}" part="{label}">\n{part}\n</document>\n\n'
                         f"Focus: {focus or 'general overview'}"),
        ]).text

    splitter = RecursiveCharacterTextSplitter(chunk_size=part_chars, chunk_overlap=0)
    parts = splitter.split_text(text)
    for _ in range(max_rounds):
        if len(parts) <= 1:
            break
        partials = [summarize(part, f"{n} of {len(parts)}") for n, part in enumerate(parts, start=1)]
        parts = splitter.split_text("\n\n".join(partials))
    return summarize("\n\n".join(parts), "full")


def _summary_part_chars() -> int:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        # ~4 chars per token; use about half the context window for the input text.
        return settings.ollama_num_ctx * 2
    return 400_000  # Claude's 1M-token context fits whole documents in one pass
