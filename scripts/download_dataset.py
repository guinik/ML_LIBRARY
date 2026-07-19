import argparse
import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["tinystories", "dailydialog"], default="tinystories")
args = parser.parse_args()

if args.dataset == "tinystories":
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir="data/tinystories", token=os.getenv("HF_TOKEN"))
else:
    print("Downloading DailyDialog...")
    ds = load_dataset("roskoN/dailydialog", cache_dir="data/dailydialog", token=os.getenv("HF_TOKEN"))

print(ds)
print("Done.")
