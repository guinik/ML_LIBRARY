import argparse
import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

DAILYDIALOG_PARQUET = {
    "train": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/roskoN/dailydialog/resolve/refs%2Fconvert%2Fparquet/full/test/0000.parquet",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinystories", "dailydialog"], default="tinystories")
args = parser.parse_args()

if args.dataset == "tinystories":
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir="data/tinystories", token=os.getenv("HF_TOKEN"))
else:
    print("Downloading DailyDialog...")
    ds = load_dataset("parquet", data_files=DAILYDIALOG_PARQUET, cache_dir="data/dailydialog", token=os.getenv("HF_TOKEN"))

print(ds)
print("Done.")
