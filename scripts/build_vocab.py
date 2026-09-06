import argparse
import array
import os
from dotenv import load_dotenv
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.normalizers import Lowercase
from tqdm import tqdm

load_dotenv()

VOCAB_SIZE   = 4096
DATA_DIR     = "data"
FLUSH_TOKENS = 1_000_000  # encode_split write-buffer size, keeps memory flat regardless of split size

PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
EOS_TOKEN  = "<EOS>"
USER_TOKEN = "user"
BOT_TOKEN  = "bot"
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

def get_text(example, dataset_name):
    if dataset_name == "tinystories":
        return example["text"]
    lines = []
    for i, utterance in enumerate(example["utterances"]):
        speaker = USER_TOKEN if i % 2 == 0 else BOT_TOKEN
        lines.append(f"{speaker} {utterance.strip()}")
    return " ".join(lines)

def text_iterator(datasets_and_names):
    for dataset, dataset_name in datasets_and_names:
        for example in tqdm(dataset, desc=f"Training tokenizer ({dataset_name})"):
            yield get_text(example, dataset_name)

def train_tokenizer(datasets_and_names, max_words):
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(
        vocab_size=max_words,
        special_tokens=SPECIAL_TOKENS,
        end_of_word_suffix="</w>",
    )
    tokenizer.train_from_iterator(text_iterator(datasets_and_names), trainer=trainer)
    return tokenizer

def encode_split(dataset, dataset_name, tokenizer, out_path):
    eos_idx = tokenizer.token_to_id(EOS_TOKEN)
    total = 0
    buf = array.array("H")
    with open(out_path, "wb") as f:
        for example in tqdm(dataset, desc=f"Encoding {out_path}"):
            buf.extend(tokenizer.encode(get_text(example, dataset_name)).ids)
            buf.append(eos_idx)
            if len(buf) >= FLUSH_TOKENS:
                f.write(buf.tobytes())
                total += len(buf)
                del buf[:]
        if buf:
            f.write(buf.tobytes())
            total += len(buf)
    print(f"  {out_path}: {total:,} tokens")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinystories", "dailydialog", "combined"], default="tinystories")
args = parser.parse_args()

if args.dataset == "combined":
    print("Loading TinyStories...")
    ds_ts = load_raw_dataset("tinystories")
    print("Loading DailyDialog...")
    ds_dd = load_raw_dataset("dailydialog")

    print("Training shared BPE tokenizer from both datasets...")
    tokenizer = train_tokenizer([(ds_ts["train"], "tinystories"), (ds_dd["train"], "dailydialog")], VOCAB_SIZE)
    tokenizer.model.save(DATA_DIR)
    print(f"  Vocab saved to {DATA_DIR}/vocab.json + {DATA_DIR}/merges.txt ({tokenizer.get_vocab_size()} tokens)")

    print("Encoding TinyStories (phase 1: pretrain)...")
    encode_split(ds_ts["train"],      "tinystories", tokenizer, f"{DATA_DIR}/train_tinystories.bin")
    encode_split(ds_ts["validation"], "tinystories", tokenizer, f"{DATA_DIR}/val_tinystories.bin")

    print("Encoding DailyDialog (phase 2: fine-tune)...")
    encode_split(ds_dd["train"],      "dailydialog", tokenizer, f"{DATA_DIR}/train_dailydialog.bin")
    encode_split(ds_dd["validation"], "dailydialog", tokenizer, f"{DATA_DIR}/val_dailydialog.bin")

    print("Done. Copy the phase-1 (tinystories) files to data/train.bin + data/val.bin to")
    print("pretrain, then copy the phase-2 (dailydialog) files over them to fine-tune.")
else:
    print(f"Loading dataset ({args.dataset})...")
    ds = load_raw_dataset(args.dataset)

    print("Training BPE tokenizer from train split...")
    tokenizer = train_tokenizer([(ds["train"], args.dataset)], VOCAB_SIZE)
    tokenizer.model.save(DATA_DIR)
    print(f"  Vocab saved to {DATA_DIR}/vocab.json + {DATA_DIR}/merges.txt ({tokenizer.get_vocab_size()} tokens)")

    print("Encoding splits...")
    encode_split(ds["train"],      args.dataset, tokenizer, f"{DATA_DIR}/train.bin")
    encode_split(ds["validation"], args.dataset, tokenizer, f"{DATA_DIR}/val.bin")

    print("Done.")
