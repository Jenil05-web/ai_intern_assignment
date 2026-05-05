"""
app.py — Streamlit chatbot UI powered by RAG + Persona + GPT-3.5-turbo

Run:
    export OPENAI_API_KEY=sk-...
    streamlit run app.py
"""

import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

from rag     import RAGIndex
from persona import extract_persona, merge_personas

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

FAISS_PREFIX = "faiss_store"
PERSONA_FILE = "persona.json"
CSV_PATH     = "conversations.csv"
NUM_CONVOS   = 50   # increase for richer index


# ── Build / Load ──────────────────────────────────────────────────────────────

def build_index_and_persona(conversations: list[str]):
    index    = RAGIndex()
    personas = []
    progress = st.progress(0, text="Building index...")

    for i, convo in enumerate(conversations):
        index.build_from_conversation(convo)
        personas.append(extract_persona(convo))
        progress.progress((i + 1) / len(conversations),
                          text=f"Processing conversation {i+1}/{len(conversations)}")

    progress.empty()
    index.save(FAISS_PREFIX)

    merged = merge_personas(personas)
    with open(PERSONA_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    return index, merged


def load_existing():
    index = RAGIndex()
    index.load(FAISS_PREFIX)
    with open(PERSONA_FILE) as f:
        persona = json.load(f)
    return index, persona


# ── Answer ────────────────────────────────────────────────────────────────────

def get_answer(question: str, index: RAGIndex, persona: dict) -> tuple[str, list]:
    rag_results = index.query(question, top_k=3)
    rag_context = "\n".join(
        f"- [{r['label']} | score {r['score']:.2f}]: {r['summary']}"
        for r in rag_results
    )

    system_prompt = """You are a helpful assistant that answers questions about a chat user.
You have their persona (habits, facts, personality, communication style) and relevant 
conversation summaries. Answer clearly and only from the provided context."""

    user_prompt = f"""Question: {question}

--- Persona ---
{json.dumps(persona, indent=2)}

--- Relevant Conversation Summaries ---
{rag_context or "No relevant summaries found."}
"""

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip(), rag_results


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="User Persona Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 User Persona RAG Chatbot")

# ── Sidebar: setup ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)

    num_convos = st.slider("Conversations to index", 10, 200, NUM_CONVOS, step=10)

    uploaded = st.file_uploader("Upload conversations.csv", type="csv")

    if st.button("🔨 Build Index", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif uploaded is None and not os.path.exists(CSV_PATH):
            st.error("Please upload conversations.csv")
        else:
            src = uploaded if uploaded else CSV_PATH
            df  = pd.read_csv(src, header=None, names=["conversation"])
            convos = df["conversation"].dropna().tolist()[:num_convos]
            with st.spinner("Building FAISS index with OpenAI embeddings..."):
                st.session_state.index, st.session_state.persona = build_index_and_persona(convos)
            st.success(f"Index built from {len(convos)} conversations!")

    if st.button("📂 Load Existing Index", use_container_width=True):
        if os.path.exists(f"{FAISS_PREFIX}.index"):
            st.session_state.index, st.session_state.persona = load_existing()
            st.success("Loaded existing index!")
        else:
            st.error("No saved index found. Build one first.")

    st.divider()
    st.caption("Suggested questions:")
    for q in ["What kind of person is this user?",
               "What are their habits?",
               "How do they talk?",
               "What topics do they discuss most?"]:
        if st.button(q, use_container_width=True):
            st.session_state.prefill = q

# ── Main: persona panel + chat ────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Extracted Persona")
    if "persona" in st.session_state:
        p = st.session_state.persona
        st.markdown("**Habits**")
        st.write(", ".join(p.get("habits", [])) or "—")
        st.markdown("**Personality**")
        st.write(", ".join(p.get("personality", [])) or "—")
        st.markdown("**Personal Facts**")
        for fact in p.get("personal_facts", []):
            st.write(f"• {fact}")
        st.markdown("**Communication Style**")
        cs = p.get("communication_style", {})
        st.json(cs)
    else:
        st.info("Build or load an index to see persona.")

with col2:
    st.subheader("💬 Chat")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 RAG Sources"):
                    for s in msg["sources"]:
                        st.write(f"**{s['label']}** (score: {s['score']:.2f})")
                        st.caption(s["summary"])

    # Prefill from sidebar buttons
    prefill = st.session_state.pop("prefill", "")

    if prompt := st.chat_input("Ask about the user...", key="chat_input") or prefill:
        if "index" not in st.session_state:
            st.warning("Please build or load an index first (sidebar).")
        else:
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Bot answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = get_answer(
                        prompt,
                        st.session_state.index,
                        st.session_state.persona,
                    )
                st.markdown(answer)
                with st.expander("📚 RAG Sources"):
                    for s in sources:
                        st.write(f"**{s['label']}** (score: {s['score']:.2f})")
                        st.caption(s["summary"])

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })