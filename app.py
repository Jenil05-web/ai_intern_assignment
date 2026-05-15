"""
app.py — Streamlit app with 3 tabs:
  Tab 1: Chat (RAG + Persona)
  Tab 2: Persona Drift Timeline
  Tab 3: Intent Classifier (offline)
  Sidebar: setup + conflict resolver
"""

import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

from rag              import RAGIndex
from persona          import extract_persona, merge_personas, detect_persona_drift
from intent_classifier import classify_intent

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

FAISS_PREFIX = "faiss_store"
PERSONA_FILE = "persona.json"
DRIFT_FILE   = "drift_timeline.json"
CSV_PATH     = "conversations.csv"
NUM_CONVOS   = 50


# ── Build / Load ──────────────────────────────────────────────────────────────

def build_index_and_persona(conversations: list[str]):
    index, personas = RAGIndex(), []
    progress = st.progress(0, text="Building index...")

    for i, convo in enumerate(conversations):
        index.build_from_conversation(convo)
        personas.append(extract_persona(convo))
        progress.progress((i + 1) / len(conversations),
                          text=f"Processing {i+1}/{len(conversations)}")

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
    results = index.query(question, top_k=3)
    context = "\n".join(f"- [{r['label']} | {r['score']:.2f}]: {r['summary']}" for r in results)
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer questions about a chat user using their persona and conversation summaries only."},
            {"role": "user",   "content": f"Question: {question}\n\nPersona:\n{json.dumps(persona, indent=2)}\n\nContext:\n{context or 'None'}"},
        ],
        max_tokens=300, temperature=0.4,
    )
    return resp.choices[0].message.content.strip(), results


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="User Persona RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 User Persona RAG Chatbot")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)

    num_convos = st.slider("Conversations to index", 10, 200, NUM_CONVOS, step=10)
    uploaded   = st.file_uploader("Upload conversations.csv", type="csv")

    if st.button("🔨 Build Index", use_container_width=True):
        if not api_key:
            st.error("Enter your OpenAI API key.")
        else:
            src   = uploaded if uploaded else CSV_PATH
            df    = pd.read_csv(src, header=None, names=["conversation"])
            convos = df["conversation"].dropna().tolist()[:num_convos]
            with st.spinner("Building FAISS index..."):
                st.session_state.index, st.session_state.persona = build_index_and_persona(convos)
            st.success(f"Done! Indexed {len(convos)} conversations.")

    if st.button("📂 Load Existing Index", use_container_width=True):
        if os.path.exists(f"{FAISS_PREFIX}.index"):
            st.session_state.index, st.session_state.persona = load_existing()
            st.success("Loaded!")
        else:
            st.error("No saved index found.")

    st.divider()

    # ── Conflict Resolver ─────────────────────────────────────────────────────
    st.subheader("🔀 Conflict Resolver")
    conflict_q = st.text_input("Ask something that might have contradictions",
                               placeholder="Did I mention my sister?")
    if st.button("Resolve", use_container_width=True):
        if "index" not in st.session_state:
            st.warning("Build or load an index first.")
        elif conflict_q:
            with st.spinner("Resolving..."):
                result = st.session_state.index.resolve_conflict(conflict_q)
            st.markdown(f"**Answer:** {result['answer']}")
            if result["contradictions_found"]:
                st.warning("⚠️ Contradictions detected across chunks")
            with st.expander("Ranked chunks"):
                for r in result["ranked_chunks"]:
                    st.write(f"**{r['label']}** — score {r['final_score']:.2f} "
                             f"(sim {r['score']:.2f} | recency {r['recency']:.2f} | emotion {r['emotion']:.2f})")
                    st.caption(r["summary"])

    st.divider()
    st.caption("Suggested questions:")
    for q in ["What kind of person is this user?", "What are their habits?",
              "How do they talk?", "What topics do they discuss most?"]:
        if st.button(q, use_container_width=True):
            st.session_state.prefill = q


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Persona Drift", "🎯 Intent Classifier"])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab1:
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
            st.json(p.get("communication_style", {}))
        else:
            st.info("Build or load an index to see persona.")

    with col2:
        st.subheader("💬 Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 RAG Sources"):
                        for s in msg["sources"]:
                            st.write(f"**{s['label']}** (score: {s['score']:.2f})")
                            st.caption(s["summary"])

        prefill = st.session_state.pop("prefill", "")
        if prompt := st.chat_input("Ask about the user...") or prefill:
            if "index" not in st.session_state:
                st.warning("Build or load an index first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer, sources = get_answer(prompt, st.session_state.index,
                                                     st.session_state.persona)
                    st.markdown(answer)
                    with st.expander("📚 RAG Sources"):
                        for s in sources:
                            st.write(f"**{s['label']}** (score: {s['score']:.2f})")
                            st.caption(s["summary"])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# ── Tab 2: Drift Timeline ─────────────────────────────────────────────────────

with tab2:
    st.subheader("📈 Persona Drift Timeline")
    st.caption("Shows how the user's tone and mood shifts across conversations.")

    drift_convos = st.slider("Conversations to analyse for drift", 5, 30, 10)
    drift_src    = uploaded if uploaded else CSV_PATH

    if st.button("▶ Run Drift Detection"):
        if not api_key:
            st.error("Enter your OpenAI API key.")
        else:
            try:
                df     = pd.read_csv(drift_src, header=None, names=["conversation"])
                convos = df["conversation"].dropna().tolist()[:drift_convos]
                with st.spinner("Analysing tone drift..."):
                    timeline = detect_persona_drift(convos)
                with open(DRIFT_FILE, "w") as f:
                    json.dump(timeline, f, indent=2)
                st.session_state.drift_timeline = timeline
            except Exception as e:
                st.error(f"Error: {e}")

    # Load existing drift if available
    if "drift_timeline" not in st.session_state and os.path.exists(DRIFT_FILE):
        with open(DRIFT_FILE) as f:
            st.session_state.drift_timeline = json.load(f)

    if "drift_timeline" in st.session_state:
        timeline = st.session_state.drift_timeline

        # Summary line
        st.markdown("### Timeline")
        for entry in timeline:
            drift_badge = "🔴 **drift**" if entry.get("drifted") else "🟢"
            st.markdown(
                f"**Day {entry['day']}** → `{entry['tone']}` & `{entry['mood']}` "
                f"— *trigger: {entry['trigger']}* {drift_badge}"
            )
            if entry.get("summary"):
                st.caption(entry["summary"])

        # Chart: mood over time
        st.markdown("### Mood over time")
        mood_map = {"happy": 5, "excited": 4, "curious": 3, "neutral": 2,
                    "anxious": 1, "frustrated": 1, "sad": 0}
        chart_data = pd.DataFrame({
            "Day":  [e["day"] for e in timeline],
            "Mood Score": [mood_map.get(e["mood"], 2) for e in timeline],
        }).set_index("Day")
        st.line_chart(chart_data)

        with st.expander("Raw JSON"):
            st.json(timeline)


# ── Tab 3: Intent Classifier ──────────────────────────────────────────────────

with tab3:
    st.subheader("🎯 Offline Intent Classifier")
    st.caption("Runs fully offline — no API calls. Classifies into: reminder / emotional_support / action_item / small_talk / unknown")

    test_msg = st.text_input("Enter a message to classify",
                             placeholder="remind me to call the doctor tomorrow")

    if st.button("Classify", use_container_width=False):
        if test_msg:
            result = classify_intent(test_msg)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Intent",      result["intent"])
            col_b.metric("Confidence",  f"{result['confidence']:.0%}")
            col_c.metric("Latency",     f"{result['latency_ms']} ms")
            st.caption(f"Method used: `{result['method']}`")

    st.markdown("---")
    st.markdown("**Batch test**")
    batch_input = st.text_area("One message per line",
                               "remind me to call mom\nI feel so lonely today\nsearch for flights to Dubai\nhaha that's so funny\nasdfg")
    if st.button("Run Batch"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        rows  = [classify_intent(l) | {"message": l} for l in lines]
        df    = pd.DataFrame(rows)[["message", "intent", "confidence", "latency_ms", "method"]]
        st.dataframe(df, use_container_width=True)