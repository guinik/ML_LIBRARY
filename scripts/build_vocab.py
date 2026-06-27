import os
import json
import re
import struct
from collections import Counter
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()

VOCAB_SIZE = 4096
DATA_DIR   = "data"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]

def tokenize(text):
    text = text.lower()
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        tok = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", tok)
        if tok:
            cleaned.append(tok)
    return cleaned

def build_vocab(dataset, max_words):
    counts = Counter()
    for example in tqdm(dataset, desc="Counting words"):
        counts.update(tokenize(example["text"]))
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, _ in counts.most_common(max_words - len(SPECIAL_TOKENS)):
        vocab[word] = len(vocab)
    return vocab

def encode_split(dataset, vocab, out_path):
    unk_idx = vocab[UNK_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    ids = []
    for example in tqdm(dataset, desc=f"Encoding {out_path}"):
        for tok in tokenize(example["text"]):
            ids.append(vocab.get(tok, unk_idx))
        ids.append(eos_idx)
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"{len(ids)}H", *ids))
    print(f"  {out_path}: {len(ids):,} tokens")

print("Loading dataset...")
ds = load_dataset("roneneldan/TinyStories", cache_dir="data/tinystories", token=os.getenv("HF_TOKEN"))

print("Building vocab from train split...")
vocab = build_vocab(ds["train"], VOCAB_SIZE)

vocab_path = f"{DATA_DIR}/vocab.json"
with open(vocab_path, "w") as f:
    json.dump(vocab, f)
print(f"  Vocab saved to {vocab_path} ({len(vocab)} words)")

print("Encoding splits...")
encode_split(ds["train"],      vocab, f"{DATA_DIR}/train.bin")
encode_split(ds["validation"], vocab, f"{DATA_DIR}/val.bin")

print("Done.")
