# Conversation RAG + Persona Chatbot

<img width="1919" height="978" alt="image" src="https://github.com/user-attachments/assets/fb75abc7-5fd0-4c71-8fee-8b133463a622" />

A system that takes a dataset of conversations, figures out what kind of person User 1 is, and lets you ask questions about them through a chatbot. Built across 4 parts — RAG pipeline, persona extraction, offline intent classifier, and conflict resolution.

---

🔗 **Live demo:** https://aiinternassignment-gatobetyhdbjgyyzftztpw.streamlit.app/

---

## What it does

**1. RAG System**
Processes conversations chronologically and creates two kinds of checkpoints — one every time the topic shifts (detected via embedding cosine similarity), and one every 100 messages. Each checkpoint gets summarised using GPT-3.5 and stored in a FAISS vector index for fast semantic search.

**2. Persona Extraction + Drift Detection**
Reads only User 1's messages and extracts habits, personality traits, personal facts, and communication style. Also tracks how their tone and mood shift across conversations over time — outputs a timeline like `Day 1 → curious & formal`, `Day 4 → casual & frustrated` along with what likely triggered each shift.

**3. Offline Intent Classifier**
A fine-tuned `bert-tiny` model (~17MB) that classifies messages into `reminder / emotional_support / action_item / small_talk / unknown`. Runs fully offline, no API calls, under 10ms per message on CPU. Train it once with `python train_intent.py`, then it works locally forever.

**4. Conflict Resolver**
Handles cases where the same topic (e.g. "my sister") appears across multiple checkpoints with contradictory context. Re-ranks retrieved chunks by recency + emotional weight, flags contradictions, and returns a single merged answer.

---

## Project structure

```
ai_intern/
├── rag.py                # checkpoints, FAISS index, topic detection, conflict resolver
├── persona.py            # persona extraction, merging, drift detection
├── intent_classifier.py  # offline inference using fine-tuned bert-tiny
├── train_intent.py       # fine-tuning script (run once)
├── app.py                # streamlit UI (3 tabs + sidebar)
├── SYSTEM_DESIGN.md      # sync architecture doc
├── SELF_EVALUATION.md    # honest self-assessment
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

You'll need an OpenAI API key — either set it as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

or paste it directly into the sidebar when the app opens.

For the intent classifier, run this once (needs internet only for the initial model download):

```bash
python train_intent.py
```

This downloads `bert-tiny`, fine-tunes it on labelled intent examples, and saves the model to `./intent_model/`. After that it's fully offline.

---

## Running it

```bash
streamlit run app.py
```

Once it opens:
1. Upload your `conversations.csv` or place it in the same folder
2. Set how many conversations to index using the slider
3. Hit **Build Index** — takes a minute depending on how many conversations you pick
4. Start chatting across 3 tabs: Chat, Persona Drift, and Intent Classifier

If you've already built the index once, click **Load Existing Index** — skips all API calls.

---

## Models used

- `text-embedding-ada-002` — embeddings for FAISS index
- `gpt-3.5-turbo` — summaries, persona extraction, chat answers
- `faiss-cpu` — local vector store
- `prajjwal1/bert-tiny` — fine-tuned offline intent classifier (~17MB)

---

## Notes

- Topic detection compares cosine similarity between consecutive 5-message chunks. When similarity drops below the threshold, a new topic starts. Not perfect but handles casual conversation well.
- `NUM_CONVOS = 50` in `app.py` is a good starting point — enough for decent coverage without burning through API credits.
- FAISS index, persona, and drift timeline all save to disk after first build, so reloads are instant with no API calls.
- The intent model is excluded from the repo (generated locally). Run `train_intent.py` to create it.
- Conflict resolver scores chunks as `0.5 × similarity + 0.3 × recency + 0.2 × emotional weight` — recency matters more than raw similarity for personal conversation data.
