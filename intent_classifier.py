"""
intent_classifier.py — Offline intent classifier using fine-tuned bert-tiny.

Loads model from ./intent_model/ (produced by train_intent.py).
Zero API calls. Runs on CPU in well under 200ms per message.

Intents: reminder / emotional_support / action_item / small_talk / unknown
"""

import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./intent_model"
MAX_LEN   = 64

# ── Load model once at import time ───────────────────────────────────────────

_tokenizer = None
_model     = None
_id2label  = None


def _load():
    global _tokenizer, _model, _id2label

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_DIR}'. Run `python train_intent.py` first."
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()

    with open(f"{MODEL_DIR}/label_map.json") as f:
        maps = json.load(f)
    _id2label = {int(k): v for k, v in maps["id2label"].items()}


def classify_intent(message: str) -> dict:
    """
    Classify a message into one of 5 intents.
    Returns {intent, confidence, latency_ms}
    """
    global _tokenizer, _model, _id2label
    if _model is None:
        _load()

    start = time.time()

    inputs = _tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs      = torch.softmax(logits, dim=1)[0]
    pred_id    = int(torch.argmax(probs))
    confidence = float(probs[pred_id])
    elapsed    = (time.time() - start) * 1000

    return {
        "intent":     _id2label[pred_id],
        "confidence": round(confidence, 3),
        "latency_ms": round(elapsed, 2),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "remind me to call the doctor tomorrow",
        "I've been feeling really anxious lately",
        "can you search for cheap flights to Dubai",
        "haha yeah that was so funny",
        "did I mention anything about my sister",
        "set a reminder for Monday",
        "I feel so overwhelmed I don't know what to do",
        "asdfghjkl",
    ]
    for t in tests:
        r = classify_intent(t)
        print(f"[{r['intent']:20s}] {r['confidence']:.0%} | {r['latency_ms']}ms — {t}")