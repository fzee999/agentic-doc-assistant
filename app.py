"""Streamlit chat UI: `streamlit run app.py`."""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from ragpipe import ingest_path, list_sources

from docagent.graph import build_graph, stream_turn
from docagent.memory import list_threads, sqlite_checkpointer

UPLOAD_DIR = Path("data/uploads")

st.set_page_config(page_title="Document Assistant", page_icon=":page_facing_up:")


@st.cache_resource
def resources():
    checkpointer = sqlite_checkpointer()
    return build_graph(checkpointer=checkpointer), checkpointer


graph, checkpointer = resources()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]

with st.sidebar:
    st.header("Conversations")
    if st.button("New conversation", width="stretch"):
        st.session_state.thread_id = uuid.uuid4().hex[:8]
    threads = list_threads(checkpointer)
    if st.session_state.thread_id not in threads:
        threads.insert(0, st.session_state.thread_id)
    st.session_state.thread_id = st.selectbox("Thread", threads, index=threads.index(st.session_state.thread_id))

    st.header("Documents")
    uploads = st.file_uploader("Add documents", type=["pdf", "md", "txt", "html"], accept_multiple_files=True)
    if uploads and st.button("Ingest uploads", width="stretch"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            path = UPLOAD_DIR / Path(upload.name).name
            path.write_bytes(upload.getvalue())
            counts = ingest_path(path)
            st.success(f"{upload.name}: {counts['chunks']} chunks indexed")
    with st.expander("Indexed documents"):
        for row in list_sources():
            st.markdown(f"- **{row['title']}**  \n`{row['source']}` ({row['chunks']} chunks)")

st.title("Document Assistant")
config = {"configurable": {"thread_id": st.session_state.thread_id}}

for message in graph.get_state(config).values.get("messages", []):
    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.text)
    elif isinstance(message, AIMessage) and message.text.strip() and not message.tool_calls:
        st.chat_message("assistant").markdown(message.text)

if question := st.chat_input("Ask about your documents"):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.status("Working...", expanded=False) as status:
            for node, step in stream_turn(graph, question, st.session_state.thread_id):
                if node == "answer":
                    answer = step
                else:
                    status.write(f"**{node}**: {step}")
            status.update(label="Agent steps", state="complete")
        st.markdown(answer)
