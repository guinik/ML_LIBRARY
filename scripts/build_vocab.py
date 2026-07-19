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

PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
EOS_TOKEN  = "<EOS>"
USER_TOKEN = "user"
BOT_TOKEN  = "bot"
# user/bot are reserved (not left to frequency ranking) so a much bigger
# tinystories corpus can never crowd them out of the vocab in combined mode
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, USER_TOKEN, BOT_TOKEN]

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
        speaker = USER_TOKEN if i % 2 == 0 else BOT_TOKEN
        lines.append(f"{speaker} {utterance.strip()}")
    return " ".join(lines)

def count_words(dataset, dataset_name, counts):
    for example in tqdm(dataset, desc=f"Counting words ({dataset_name})"):
        counts.update(tokenize(get_text(example, dataset_name)))

def build_vocab_from_counts(counts, max_words):
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    reserved = set(SPECIAL_TOKENS)
    for word, _ in counts.most_common():
        if len(vocab) >= max_words:
            break
        if word in reserved:
            continue
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

def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)
    print(f"  Vocab saved to {path} ({len(vocab)} words)")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinystories", "dailydialog", "combined"], default="tinystories")
args = parser.parse_args()

vocab_path = f"{DATA_DIR}/vocab.json"

if args.dataset == "combined":
    print("Loading TinyStories...")
    ds_ts = load_raw_dataset("tinystories")
    print("Loading DailyDialog...")
    ds_dd = load_raw_dataset("dailydialog")

    print("Building shared vocab from both datasets (user/bot reserved)...")
    counts = Counter()
    count_words(ds_ts["train"], "tinystories", counts)
    count_words(ds_dd["train"], "dailydialog", counts)
    vocab = build_vocab_from_counts(counts, VOCAB_SIZE)
    save_vocab(vocab, vocab_path)

    print("Encoding TinyStories (phase 1: pretrain)...")
    encode_split(ds_ts["train"],      "tinystories", vocab, f"{DATA_DIR}/train_tinystories.bin")
    encode_split(ds_ts["validation"], "tinystories", vocab, f"{DATA_DIR}/val_tinystories.bin")

    print("Encoding DailyDialog (phase 2: fine-tune)...")
    encode_split(ds_dd["train"],      "dailydialog", vocab, f"{DATA_DIR}/train_dailydialog.bin")
    encode_split(ds_dd["validation"], "dailydialog", vocab, f"{DATA_DIR}/val_dailydialog.bin")

    print("Done. Copy the phase-1 (tinystories) files to data/train.bin + data/val.bin to")
    print("pretrain, then copy the phase-2 (dailydialog) files over them to fine-tune.")
else:
    print(f"Loading dataset ({args.dataset})...")
    ds = load_raw_dataset(args.dataset)

    print("Building vocab from train split...")
    counts = Counter()
    count_words(ds["train"], args.dataset, counts)
    vocab = build_vocab_from_counts(counts, VOCAB_SIZE)
    save_vocab(vocab, vocab_path)

    print("Encoding splits...")
    encode_split(ds["train"],      args.dataset, vocab, f"{DATA_DIR}/train.bin")
    encode_split(ds["validation"], args.dataset, vocab, f"{DATA_DIR}/val.bin")

    print("Done.")
