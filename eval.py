"""Held-out perplexity on val.bin.

    python eval.py --ckpt /ckpt/best.pt
"""
import os
import sys
import math
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, TrainConfig  # noqa
from sample import load_model               # noqa


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(TrainConfig().ckpt_dir, "best.pt"))
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg = load_model(args.ckpt, device)
    val = np.memmap(DataConfig().val_bin, dtype=np.uint16, mode="r")
    T = mcfg.seq_len

    losses = []
    for _ in range(args.iters):
        ix = torch.randint(len(val) - T - 1, (args.batch,))
        x = torch.stack([torch.from_numpy(val[i:i+T].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(val[i+1:i+1+T].astype(np.int64)) for i in ix]).to(device)
        tgt = y.unsqueeze(1)          # (B,1,T): score the main (t+1) head only
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
                else torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss, _ = model(x, tgt)
        losses.append(loss.item())
    m = sum(losses) / len(losses)
    print(f"val loss {m:.4f} | perplexity {math.exp(m):.2f} (over {args.iters} batches)")


if __name__ == "__main__":
    main()
