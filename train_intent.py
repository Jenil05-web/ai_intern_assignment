"""
train_intent.py — Fine-tune a lightweight BERT model for intent classification.

Primary:  downloads prajjwal1/bert-tiny (~17MB) from HuggingFace — run once with internet.
Fallback: builds the same architecture locally if HuggingFace is unreachable.

Intents: reminder / emotional_support / action_item / small_talk / unknown

Run:
    python train_intent.py

Saves model to ./intent_model/ — after that, intent_classifier.py works fully offline.
"""

import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_DIR = "./intent_model"
MAX_LEN    = 64
BATCH_SIZE = 16
EPOCHS     = 6
LR         = 3e-4
SEED       = 42

LABELS   = ["reminder", "emotional_support", "action_item", "small_talk", "unknown"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

RAW_DATA = {
    "reminder": [
        "remind me to call mom tomorrow",
        "don't forget to submit the report by Friday",
        "set a reminder for my dentist appointment",
        "I need to remember to pay rent this month",
        "can you remind me about the meeting on Monday",
        "note that I have to pick up the kids at 3pm",
        "remind me later to take my medicine",
        "don't let me forget to reply to that email",
        "I should remember to water the plants today",
        "remind me to charge my laptop before the trip",
        "set an alarm for 7am tomorrow morning",
        "I need a reminder to call the bank",
        "note to self buy groceries on the way home",
        "can you ping me about the deadline next week",
        "don't forget we have a call at noon",
        "add a reminder for the team standup at 10",
        "I keep forgetting to send that invoice",
        "put a note to follow up with the client tomorrow",
        "remind me to check my emails tonight",
        "I must not forget to book the flight tickets",
        "schedule a reminder for my weekly review",
        "remind me to wish Sarah happy birthday",
        "I need to remember to renew my passport",
        "set a reminder to review the contract",
        "don't forget to back up my files tonight",
    ],
    "emotional_support": [
        "I am feeling really sad today",
        "nobody understands what I am going through",
        "I have been so anxious lately and can not stop worrying",
        "I feel completely overwhelmed and do not know what to do",
        "I am so stressed I can barely function anymore",
        "I just need someone to talk to right now",
        "everything feels hopeless and I do not know why",
        "I had a terrible day and just want to vent",
        "I feel so alone even when I am with people",
        "I am really struggling and could use some support",
        "I do not think anyone cares about me at all",
        "I have been crying all day and can not stop",
        "I feel like I am failing at everything in my life",
        "I am scared about what is going to happen next",
        "this is too much for me to handle on my own",
        "I just feel numb and empty inside lately",
        "I am really not okay right now",
        "I feel like giving up on everything",
        "I am so frustrated and do not know who to talk to",
        "life feels really difficult right now and I need help",
        "I feel like nobody listens to me",
        "I am having a really hard time coping",
        "I feel so lost and confused about everything",
        "I just want to feel better but nothing helps",
        "I can not sleep because I keep worrying about things",
    ],
    "action_item": [
        "can you search for flights to London next week",
        "book a table at an Italian restaurant for tonight",
        "send an email to the team about the project update",
        "create a task for the new feature in the backlog",
        "help me draft a message to my manager about the issue",
        "find me a good recipe for pasta carbonara",
        "look up the weather in New York for tomorrow",
        "add this item to my to-do list please",
        "write a summary of the meeting notes from today",
        "translate this paragraph to Spanish for me",
        "calculate the total cost including tax and shipping",
        "order pizza to my home address tonight",
        "find nearby coffee shops that are open right now",
        "help me write a cover letter for this job",
        "make a list of things I need to pack for the trip",
        "schedule a meeting with the design team this week",
        "pull up the latest sales report from last quarter",
        "convert 100 dollars to euros at current rate",
        "find the phone number for the nearest hospital",
        "create a new document and share it with Sarah",
        "help me plan a budget for next month",
        "search for the best laptop under 1000 dollars",
        "write a professional bio for my LinkedIn profile",
        "find a good hotel near the conference venue",
        "set up a new project folder on my desktop",
    ],
    "small_talk": [
        "how are you doing today",
        "that is so funny haha",
        "what do you think about the weather lately",
        "good morning hope you have a great day",
        "I love pizza what is your favourite food",
        "lol that totally made my day better",
        "nice talking to you as always",
        "what is your favourite movie of all time",
        "I have been watching a lot of Netflix lately",
        "did you see the game last night it was amazing",
        "it is such a beautiful day outside today",
        "I am just chilling at home today relaxing",
        "haha yeah that makes total sense to me",
        "I can not believe how fast time flies these days",
        "weekends are always way too short",
        "just had the best coffee I have ever tasted",
        "I am binge watching a really good new show",
        "wow that is actually really interesting",
        "I did not know that thanks for sharing with me",
        "you always manage to make me laugh",
        "what are you up to this weekend",
        "I had such a relaxing morning today",
        "the weather has been so nice lately",
        "I really enjoy our conversations",
        "that movie was incredible I loved every minute",
    ],
    "unknown": [
        "asdfghjkl",
        "hmm",
        "ok",
        "maybe I guess",
        "I do not know",
        "42",
        "...",
        "whatever",
        "sure",
        "yeah no",
        "mmk",
        "idk lol",
        "uhh",
        "nah",
        "meh",
        "blah blah blah",
        "testing 123",
        "hello world",
        "xyz abc",
        "random text here nothing important at all",
        "qwerty",
        "just some words",
        "nothing specific",
        "no idea",
        "skip this one",
    ],
}


def augment(text: str) -> list[str]:
    """Simple word-swap augmentation."""
    swaps = {
        "remind":   "remember",
        "book":     "reserve",
        "search":   "look up",
        "feel":     "am feeling",
        "sad":      "down",
        "anxious":  "nervous",
        "send":     "dispatch",
        "create":   "make",
        "funny":    "hilarious",
        "stressed": "overwhelmed",
        "great":    "wonderful",
        "terrible": "awful",
    }
    variants = [text]
    for word, rep in swaps.items():
        if word in text.lower() and len(variants) < 3:
            variants.append(text.lower().replace(word, rep))
    return variants


def build_dataset():
    texts, labels = [], []
    for label, examples in RAW_DATA.items():
        for ex in examples:
            for aug in augment(ex):
                texts.append(aug)
                labels.append(LABEL2ID[label])
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Model loader (HuggingFace or local fallback) ──────────────────────────────

def load_model_and_tokenizer():
    """
    Try to load prajjwal1/bert-tiny from HuggingFace.
    If offline/blocked, build the same tiny BERT architecture from scratch
    using bert-base-uncased tokenizer config (also cached locally).
    """
    try:
        print(f"Trying to load {MODEL_NAME} from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        print("Loaded from HuggingFace.")
        return tokenizer, model

    except Exception as e:
        print(f"HuggingFace unavailable ({e.__class__.__name__}). Building model locally...")

        # bert-tiny architecture: 2 layers, 128 hidden, 2 heads
        config = BertConfig(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
            type_vocab_size=2,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = BertForSequenceClassification(config)

        # Build a simple word-piece tokenizer using basic vocab
        # Use bert-base-uncased tokenizer if cached, else a simple whitespace fallback
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            print("Using cached bert-base-uncased tokenizer.")
        except Exception:
            # Absolute fallback: build minimal tokenizer
            from transformers import BasicTokenizer
            print("Using basic whitespace tokenizer fallback.")
            # We'll create a minimal wrapper
            tokenizer = _build_minimal_tokenizer()

        return tokenizer, model


class _MinimalTokenizer:
    """Dead-simple whitespace tokenizer fallback — last resort only."""
    VOCAB_SIZE = 30522

    def __init__(self):
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def __call__(self, texts, truncation=True, padding=True, max_length=64, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        all_ids, all_masks = [], []
        for t in texts:
            tokens = [self.cls_token_id]
            for word in t.lower().split():
                tokens.append(hash(word) % (self.VOCAB_SIZE - 200) + 100)
            tokens.append(self.sep_token_id)
            tokens = tokens[:max_length]
            mask   = [1] * len(tokens)
            while len(tokens) < max_length:
                tokens.append(self.pad_token_id)
                mask.append(0)
            all_ids.append(tokens)
            all_masks.append(mask)
        return {
            "input_ids":      torch.tensor(all_ids),
            "attention_mask": torch.tensor(all_masks),
        }

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/minimal_tokenizer.json", "w") as f:
            json.dump({"type": "minimal"}, f)


def _build_minimal_tokenizer():
    return _MinimalTokenizer()


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    tokenizer, model = load_model_and_tokenizer()

    texts, labels = build_dataset()
    print(f"Dataset: {len(texts)} examples | Classes: {LABELS}")

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED, stratify=labels
    )

    train_ds = IntentDataset(X_train, y_train, tokenizer)
    val_ds   = IntentDataset(X_val,   y_val,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, len(train_loader) // 2),
        num_training_steps=len(train_loader) * EPOCHS,
    )

    device = torch.device("cpu")
    model.to(device)

    best_acc   = 0.0
    best_state = None

    print("\nTraining bert-tiny on CPU...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                out  = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                )
                preds.extend(torch.argmax(out.logits, dim=1).tolist())
                trues.extend(batch["labels"].tolist())

        acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss: {total_loss/len(train_loader):.4f}  val_acc: {acc:.2%}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    model.load_state_dict(best_state)
    print(f"\nBest val accuracy: {best_acc:.2%}")
    print("\nClassification Report (last epoch):")
    print(classification_report(trues, preds, target_names=LABELS, zero_division=0))

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    try:
        tokenizer.save_pretrained(OUTPUT_DIR)
    except Exception:
        pass   # minimal tokenizer saves its own way

    with open(f"{OUTPUT_DIR}/label_map.json", "w") as f:
        json.dump({"id2label": {str(k): v for k, v in ID2LABEL.items()},
                   "label2id": LABEL2ID}, f)

    size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        for fn in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, fn))
    ) / (1024 * 1024)

    print(f"\nSaved to {OUTPUT_DIR}/  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    train()