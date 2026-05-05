# Conversation RAG + Persona Chatbot

A small AI system I built as part of an internship assignment. The idea was to take a dataset of conversations, understand what kind of person User 1 is, and then be able to answer questions about them using a chatbot.

---

## What it does

There are three main parts:

**1. RAG System**
Processes conversations chronologically and creates two kinds of checkpoints — one every time the topic shifts, and one every 100 messages. Each checkpoint gets summarised using GPT-3.5. All the summaries are stored in a FAISS vector index so we can do fast semantic search later.

**2. Persona Extraction**
Looks at only User 1's messages and extracts things like their habits, personality traits, personal facts, and how they communicate. This gets saved as a JSON file.

**3. Streamlit Chatbot**
A simple UI where you can ask things like "what are this user's habits?" or "how do they talk?" — it pulls from both the persona JSON and the FAISS index to give a grounded answer.

---

## Project structure

```
ai_intern/
├── rag.py              # checkpoints, FAISS index, topic detection
├── persona.py          # persona extraction + merging
├── app.py              # streamlit UI
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

You'll need an OpenAI API key. You can either set it as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

or just paste it directly into the sidebar when the app opens.

---

## Running it

```bash
streamlit run app.py
```

Once it opens:
1. Upload your `conversations.csv` (or place it in the same folder)
2. Set how many conversations to index using the slider
3. Hit **Build Index** — this will take a minute depending on how many convos you pick
4. Start chatting

If you've already built the index once, just click **Load Existing Index** and it'll skip the API calls.

---

## Models used

- `text-embedding-ada-002` — for embeddings
- `gpt-3.5-turbo` — for summaries and chat answers
- `faiss-cpu` — vector store

---

## Notes

- The topic detection works by comparing embedding similarity between consecutive message chunks. When the similarity drops below a threshold, it marks a new topic. It's not perfect but works well enough for casual conversations.
- Keeping `NUM_CONVOS` around 50 is a good balance between quality and API cost. Crank it up if you want richer results.
- The FAISS index and persona get saved to disk after the first build, so subsequent loads are instant.
