import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

print("Downloading TinyStories...")
ds = load_dataset("roneneldan/TinyStories", cache_dir="data/tinystories", token=os.getenv("HF_TOKEN"))
print(ds)
print("Done.")
