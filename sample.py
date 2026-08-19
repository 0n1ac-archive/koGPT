"""Generate text from a trained checkpoint.

    python sample.py --ckpt /ckpt/best.pt --prompt "인공지능은"
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TokenizerConfig, get_model_config  # noqa
from model import GPT                                  # noqa


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = get_model_config()
    mcfg.__dict__.update(ck["model_config"])
    model = GPT(mcfg)
    # strip a possible torch.compile prefix
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model.to(device).eval(), mcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.environ.get("KOGPT_CKPT_DIR", "/ckpt"), "best.pt"))
    ap.add_argument("--prompt", default="인공지능은")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=TokenizerConfig().model_file)
    model, _ = load_model(args.ckpt, device)

    ids = [1] + sp.encode(args.prompt, out_type=int)   # bos + prompt
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for i in range(args.n):
        out = model.generate(x, args.tokens, args.temperature, args.top_k)[0].tolist()
        print(f"\n===== sample {i+1} =====")
        print(sp.decode(out))


if __name__ == "__main__":
    main()
