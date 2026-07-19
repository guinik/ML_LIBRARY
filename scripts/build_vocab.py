import argparse
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

DAILYDIALOG_PARQUET = {
    "train": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet",
}

def load_raw_dataset(dataset_name):
    if dataset_name == "tinystories":
        return load_dataset("roneneldan/TinyStories", cache_dir="data/tinystories", token=os.getenv("HF_TOKEN"))
    return load_dataset("parquet", data_files=DAILYDIALOG_PARQUET, cache_dir="data/dailydialog", token=os.getenv("HF_TOKEN"))

def tokenize(text):
    text = text.lower()
    tokens = text.split()
    cleaned = []
    for tok in tokens:
        tok = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", tok)
        if tok:
            cleaned.append(tok)
    return cleaned

def get_text(example, dataset_name):
    if dataset_name == "tinystories":
        return example["text"]
    lines = []
    for i, utterance in enumerate(example["utterances"]):
        speaker = "user" if i % 2 == 0 else "bot"
        lines.append(f"{speaker} {utterance.strip()}")
    return " ".join(lines)

def build_vocab(dataset, dataset_name, max_words):
    counts = Counter()
    for example in tqdm(dataset, desc="Counting words"):
        counts.update(tokenize(get_text(example, dataset_name)))
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, _ in counts.most_common(max_words - len(SPECIAL_TOKENS)):
        vocab[word] = len(vocab)
    return vocab

def encode_split(dataset, dataset_name, vocab, out_path):
    unk_idx = vocab[UNK_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    ids = []
    for example in tqdm(dataset, desc=f"Encoding {out_path}"):
        for tok in tokenize(get_text(example, dataset_name)):
            ids.append(vocab.get(tok, unk_idx))
        ids.append(eos_idx)
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"{len(ids)}H", *ids))
    print(f"  {out_path}: {len(ids):,} tokens")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinystories", "dailydialog"], default="tinystories")
args = parser.parse_args()

print(f"Loading dataset ({args.dataset})...")
ds = load_raw_dataset(args.dataset)

print("Building vocab from train split...")
vocab = build_vocab(ds["train"], args.dataset, VOCAB_SIZE)

vocab_path = f"{DATA_DIR}/vocab.json"
with open(vocab_path, "w") as f:
    json.dump(vocab, f)
print(f"  Vocab saved to {vocab_path} ({len(vocab)} words)")

print("Encoding splits...")
encode_split(ds["train"],      args.dataset, vocab, f"{DATA_DIR}/train.bin")
encode_split(ds["validation"], args.dataset, vocab, f"{DATA_DIR}/val.bin")

print("Done.")
