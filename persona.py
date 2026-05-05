"""
persona.py — Extracts structured persona from User 1 messages using GPT-3.5-turbo.
Merges results across multiple conversations.
"""

import os
import re
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _to_list(val) -> list:
    """Safely coerce any GPT output to a flat list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v is not None]
    if isinstance(val, dict):
        # e.g. {"coffee": true, "reader": false} → ["coffee"]
        return [k for k, v in val.items() if v]
    if isinstance(val, str):
        return [val]
    return []


def extract_persona(raw_text: str) -> dict:
    """
    Collect all User 1 lines, send to GPT-3.5, get back structured JSON persona.
    """
    user1_lines = [
        line[7:].strip()
        for line in raw_text.strip().split("\n")
        if line.strip().startswith("User 1:")
    ]
    if not user1_lines:
        return {}

    sample = "\n".join(user1_lines[:40])

    prompt = f"""Analyse these chat messages and extract a persona. 
Return ONLY a valid JSON object — no markdown, no explanation — with exactly these 4 keys:

{{
  "habits": ["list", "of", "habits"],
  "personal_facts": ["list", "of", "facts"],
  "personality": ["list", "of", "traits"],
  "communication_style": {{
    "message_length": "short or medium or long",
    "tone": "casual or formal",
    "emoji_usage": "none or low or high",
    "asks_questions": true or false
  }}
}}

User messages:
{sample}
"""

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    # Strip ```json ... ``` if present
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {}

    # Safely normalise every field so merge_personas never crashes
    return {
        "habits":          _to_list(data.get("habits")),
        "personal_facts":  _to_list(data.get("personal_facts")),
        "personality":     _to_list(data.get("personality")),
        "communication_style": {
            "message_length": str(data.get("communication_style", {}).get("message_length", "unknown")),
            "tone":           str(data.get("communication_style", {}).get("tone", "unknown")),
            "emoji_usage":    str(data.get("communication_style", {}).get("emoji_usage", "unknown")),
            "asks_questions": bool(data.get("communication_style", {}).get("asks_questions", False)),
        },
    }


def merge_personas(personas: list[dict]) -> dict:
    """Aggregate persona dicts from multiple conversations."""
    from collections import Counter

    habit_ctr    = Counter()
    fact_set     = set()
    pers_ctr     = Counter()
    lengths, emojis, tones, questions = [], [], [], []

    for p in personas:
        if not p:
            continue
        habit_ctr.update(_to_list(p.get("habits")))
        fact_set.update(_to_list(p.get("personal_facts")))
        pers_ctr.update(_to_list(p.get("personality")))
        cs = p.get("communication_style") or {}
        if cs.get("message_length"): lengths.append(cs["message_length"])
        if cs.get("emoji_usage"):    emojis.append(cs["emoji_usage"])
        if cs.get("tone"):           tones.append(cs["tone"])
        if "asks_questions" in cs:   questions.append(cs["asks_questions"])

    def most_common(lst):
        return Counter(lst).most_common(1)[0][0] if lst else "unknown"

    return {
        "habits":          [h for h, _ in habit_ctr.most_common(5)],
        "personal_facts":  list(fact_set)[:10],
        "personality":     [t for t, _ in pers_ctr.most_common(5)],
        "communication_style": {
            "message_length": most_common(lengths),
            "tone":           most_common(tones),
            "emoji_usage":    most_common(emojis),
            "asks_questions": most_common(questions),
        },
    }