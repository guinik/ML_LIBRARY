import time

import torch
import torch.nn as nn

# Same architecture/hyperparameters as src/benchmark.cpp, so timings are comparable.
VOCAB_SIZE   = 4096
SEQ_LEN      = 64
EMBED_DIM    = 256
D_K          = 256
NUM_LAYERS   = 6
BATCH_SIZE   = 16
LR           = 3e-4
WARMUP_STEPS = 10
BENCH_STEPS  = 50


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, causal):
        super().__init__()
        self.wq = nn.Linear(d_model, d_k, bias=False)
        self.wk = nn.Linear(d_model, d_k, bias=False)
        self.wv = nn.Linear(d_model, d_k, bias=False)
        self.wo = nn.Linear(d_k, d_model, bias=False)
        self.d_k = d_k
        self.causal = causal

    def forward(self, x):
        Q, K, V = self.wq(x), self.wk(x), self.wv(x)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if self.causal:
            seq = x.size(1)
            mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return self.wo(attn @ V)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, d_k, causal):
        super().__init__()
        self.attn = SingleHeadAttention(embed_dim, d_k, causal)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        normed = self.ln1(x + self.attn(x))
        return self.ln2(normed + self.ffn(normed))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_k, num_layers, causal=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(512, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, d_k, causal) for _ in range(num_layers)])
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = TinyTransformer(VOCAB_SIZE, EMBED_DIM, D_K, NUM_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(WARMUP_STEPS):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_STEPS):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed * 1000 / BENCH_STEPS
    tokens_per_sec = BATCH_SIZE * SEQ_LEN * BENCH_STEPS / elapsed

    print(f"Steps: {BENCH_STEPS}")
    print(f"Total time: {elapsed:.3f} s")
    print(f"Time/step: {ms_per_step:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:,.0f}")


if __name__ == "__main__":
    main()
