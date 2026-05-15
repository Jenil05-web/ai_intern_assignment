"""
intent_classifier.py — Offline intent classifier using zero-shot with a tiny model.
Model: typeform/distilbart-mnli-12-3 (~250MB) OR a keyword fallback that runs in <5ms.

Since the constraint is <50MB + CPU <200ms, we use a two-layer approach:
  Layer 1: Rule-based keyword matching (instant, covers ~80% of cases)
  Layer 2: TF-IDF + cosine similarity against intent templates (no model download needed)

No OpenAI. No internet required at runtime.
"""

import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Intent templates (training examples per class) ────────────────────────────
# These act as our "fine-tuned" examples via TF-IDF similarity

INTENT_EXAMPLES = {
    "reminder": [
        "remind me to call mom tomorrow",
        "don't forget to submit the report",
        "set a reminder for Monday meeting",
        "I need to remember to buy groceries",
        "can you remind me about the dentist appointment",
        "note that I have to pay rent on friday",
        "remind me later about this",
    ],
    "emotional_support": [
        "I am feeling really sad today",
        "I don't know what to do I'm so stressed",
        "nobody understands me",
        "I've been anxious all week",
        "I feel so lonely lately",
        "I'm overwhelmed and can't cope",
        "everything feels pointless right now",
        "I had a terrible day and just need to vent",
    ],
    "action_item": [
        "can you search for flights to London",
        "book a table at the restaurant",
        "send an email to the team about the update",
        "create a task for the project",
        "add this to my to-do list",
        "help me draft a message to my boss",
        "find me a recipe for pasta",
    ],
    "small_talk": [
        "how are you doing today",
        "what do you think about the weather",
        "that's so funny lol",
        "I love pizza what about you",
        "good morning hope your day is well",
        "haha yeah that makes sense",
        "nice talking to you",
        "what's your favourite movie",
    ],
    "unknown": [
        "asdfgh",
        "hmm",
        "ok",
        "maybe",
        "I don't know",
        "42",
        "...",
    ],
}


KEYWORD_RULES = {
    "reminder": [
        r"\b(remind|reminder|don'?t forget|remember to|note that|schedule|set an? alarm)\b"
    ],
    "emotional_support": [
        r"\b(sad|depressed|anxious|stressed|overwhelmed|lonely|scared|hopeless|crying|upset|miserable|frustrated)\b",
        r"\b(i feel|i'm feeling|feeling so|can'?t cope|need support|need to vent)\b",
    ],
    "action_item": [
        r"\b(search for|look up|find me|book|send|create|add to|draft|write|buy|order|schedule|set up)\b"
    ],
    "small_talk": [
        r"\b(haha|lol|lmao|how are you|good morning|good night|what'?s up|nice to|love (to|talking)|funny)\b"
    ],
}


class IntentClassifier:
    def __init__(self):
        # Flatten examples for TF-IDF
        self.labels   = []
        self.examples = []
        for intent, texts in INTENT_EXAMPLES.items():
            for t in texts:
                self.labels.append(intent)
                self.examples.append(t)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self.vectors    = self.vectorizer.fit_transform(self.examples)

    def _keyword_match(self, text: str) -> str | None:
        text = text.lower()
        for intent, patterns in KEYWORD_RULES.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        return None

    def classify(self, message: str) -> dict:
        start = time.time()

        keyword_result = self._keyword_match(message)
        if keyword_result:
            elapsed = (time.time() - start) * 1000
            return {
                "intent":     keyword_result,
                "confidence": 0.90,
                "method":     "keyword",
                "latency_ms": round(elapsed, 2),
            }

        msg_vec = self.vectorizer.transform([message])
        sims    = cosine_similarity(msg_vec, self.vectors)[0]
        top_idx = int(np.argmax(sims))
        score   = float(sims[top_idx])

        elapsed = (time.time() - start) * 1000
        return {
            "intent":     self.labels[top_idx] if score > 0.15 else "unknown",
            "confidence": round(score, 3),
            "method":     "tfidf",
            "latency_ms": round(elapsed, 2),
        }


_classifier = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(message: str) -> dict:
    """Public interface. Returns {intent, confidence, method, latency_ms}."""
    return get_classifier().classify(message)


if __name__ == "__main__":
    tests = [
        "remind me to call the doctor tomorrow",
        "I've been feeling really anxious lately",
        "can you search for cheap flights to Dubai",
        "haha yeah that was so funny",
        "did I mention anything about my sister",
        "set a reminder for Monday",
        "I feel so overwhelmed I don't know what to do",
    ]
    clf = IntentClassifier()
    for t in tests:
        r = clf.classify(t)
        print(f"[{r['intent']:18s}] ({r['confidence']:.2f} | {r['latency_ms']}ms) — {t}")