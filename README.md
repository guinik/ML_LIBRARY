# ML_LIBRARY

A from-scratch C++20 tensor/autograd library with optional CUDA acceleration, used here to train a small causal Transformer language model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Everything — tensors, the autograd graph, layers, the Adam optimizer, and a custom checkpoint format — is implemented in-house in `include/` and `src/`. There are no ML framework dependencies (no PyTorch/TensorFlow/libtorch). CUDA kernels (`src/Cuda*.cu`) are an optional accelerated backend behind the same API.

## Project layout

```
include/, src/          Core library: Tensor, Node/ExecutionGraph (autograd), Layers,
                         AdamOptimizer, TransformerMiniModel, Serialization, DataLoader,
                         and Cuda* files for the optional GPU backend.
src/main.cpp             Fixed training entry point (trains TransformerMiniModel on TinyStories).
scripts/download_dataset.py  Downloads TinyStories via HuggingFace `datasets`.
scripts/build_vocab.py       Builds a word-level vocab and encodes train/val splits to .bin token files.
CMakeLists.txt           Build definition; `USE_CUDA` option toggles the cuBLAS/CUDA backend.
```

Data (`data/`) and checkpoints (`*.mlt`) are gitignored — you generate/produce them locally.

## Requirements

- CMake ≥ 3.20 and a C++20 compiler (MSVC, GCC, or Clang)
- Optional: CUDA Toolkit (for `-DUSE_CUDA=ON`, uses cuBLAS + custom kernels)
- Python 3 with `requirements.txt` (`datasets`, `python-dotenv`, `tqdm`) for data preparation only — not needed to build/run the C++ code

## 1. Prepare the data

From the repo root:

```bash
pip install -r requirements.txt
python scripts/download_dataset.py   # caches TinyStories under data/tinystories
python scripts/build_vocab.py        # writes data/vocab.json, data/train.bin, data/val.bin
```

`build_vocab.py` builds a 4096-word vocabulary (with `<PAD>`, `<UNK>`, `<EOS>`) and encodes each split into a flat binary file of `uint16` token IDs, matching what `DataLoader` expects.

TinyStories is a public HF dataset, so no token is required. If you hit rate limits, copy `.env.example` to `.env` and set `HF_TOKEN`.

## 2. Build

CPU only:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

With CUDA acceleration:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces the `EXECUTE` binary in `build/`.

## 3. Train

`src/main.cpp` is a fixed (non-CLI) training script — hyperparameters are constants at the top of the file (`VOCAB_SIZE=4096`, `SEQ_LEN=64`, `EMBED_DIM=256`, `NUM_LAYERS=6`, `BATCH_SIZE=16`, `STEPS=100000`, cosine LR schedule with warmup). Edit and rebuild if you want to change them.

Run it **from the `build/` directory** — the paths `../data/train.bin`, `../data/vocab.json`, and `../tinystories.mlt` are relative to the executable's working directory, so it expects `build/` to sit directly under the repo root:

```bash
cd build
./EXECUTE
```

It will:
- Print the parameter count and resume from `../tinystories.mlt` automatically if that file already exists.
- Log loss/LR/ETA every 100 steps.
- Save a checkpoint to `../tinystories.mlt` and print a sample generation every 1000 steps.
- Save a final checkpoint and generate a sample at the end of training.

## Checkpoint format (`.mlt`)

A small custom binary format (see `include/Serialization.hpp`): magic bytes, version, a JSON header describing each tensor's name/shape/byte offsets, padding to a 64-byte boundary, then the raw `float32` tensor data. `model.save(path)` / `model.load(path)` read and write this directly — no external serialization library involved.

---

## Running training on Google Colab

Colab gives you a free GPU, which the CUDA backend can use via cuBLAS. Steps below assume the repo is on GitHub (`origin` → `https://github.com/guinik/ML_LIBRARY`).

**1. Set the runtime to GPU:** Runtime → Change runtime type → T4 GPU (or better).

**2. Clone the repo and check the toolchain:**

```python
!git clone https://github.com/guinik/ML_LIBRARY.git
%cd ML_LIBRARY
!nvcc --version
!cmake --version
```

Colab's default image already has a CUDA toolkit and a compatible g++, so `nvcc --version` should just work.

**3. Prepare the data:**

```python
!pip install -q -r requirements.txt
!python scripts/download_dataset.py
!python scripts/build_vocab.py
```

This takes a few minutes the first time (downloads + tokenizes TinyStories). `data/` lands under `/content/ML_LIBRARY/data`.

**4. Configure and build with CUDA:**

```python
!cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j$(nproc)
```

**5. Train:**

```python
%cd build
!./EXECUTE
```

Since `main.cpp` is a plain blocking loop that logs to stdout, running it in a Colab cell streams the log live in the cell output. `STEPS=100000` is a lot — expect this to run for a long while on a T4; feel free to lower `STEPS` in `src/main.cpp` and rebuild (step 4) if you just want to verify things work end-to-end first.

**Important — Colab's disk is ephemeral.** The VM (and everything in `/content`) is wiped when the runtime disconnects or recycles, so `tinystories.mlt` will be lost unless you copy it somewhere durable. Since checkpoints are written every 1000 steps to `../tinystories.mlt` (i.e. `/content/ML_LIBRARY/tinystories.mlt`), mount Drive first and periodically copy it out — e.g. in a second cell while training runs in the first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!cp /content/ML_LIBRARY/tinystories.mlt /content/drive/MyDrive/tinystories.mlt
```

If a session disconnects mid-run, just re-clone/build and copy the checkpoint back from Drive into the repo root before re-running `./EXECUTE` — it auto-resumes from `../tinystories.mlt` if present.

That covers running training. Saving the final model off Colab (via the same `cp` to Drive, or `files.download(...)`) and serving it are separate next steps once training is where you want it.

## Benchmark: ML_LIBRARY vs PyTorch

`src/benchmark.cpp` (target `BENCHMARK`) and `scripts/benchmark_pytorch.py` run the *same* model — same architecture (embedding + learned positional embedding, 6 single-head causal attention blocks with post-attention/post-FFN LayerNorm, `embed_dim × 4` ReLU FFN, `4096`-vocab output head), same dimensions (`embed_dim=256`, `seq_len=64`, `batch=16`), and the same Adam hyperparameters (`lr=3e-4`, `β1=0.9`, `β2=0.999`, `eps=1e-8`). Both use synthetic random token IDs (no data loading) so the timing only reflects forward + backward + optimizer-step compute, with 10 untimed warmup steps followed by 50 timed steps and an explicit device sync before/after timing.

Build and run the C++ side (CUDA build, from `build/`):

```bash
cmake --build build -j
cd build
./BENCHMARK
```

Run the PyTorch side (Colab already has `torch` installed):

```bash
python scripts/benchmark_pytorch.py
```

Each prints parameter count, total time, ms/step, and tokens/sec — compare those numbers directly. Since both use the same seed data, this isolates "how fast is the custom C++/CUDA kernel + autograd stack" vs. "how fast is PyTorch/cuBLAS" for this exact model shape, without the data pipeline or checkpointing in the way.
