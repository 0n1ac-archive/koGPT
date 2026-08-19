"""Tiny end-to-end sanity check (CPU-friendly). Runs both attention types,
MTP loss, a backward step, and generation. Catches shape/logic bugs before we
spend GPU-hours.  Run:  .venv/bin/python smoke_test.py
"""
import torch
from config import ModelConfig
from model import GPT


def run(attn_type, mtp):
    cfg = ModelConfig(vocab_size=512, n_layer=2, n_embd=128, head_dim=32,
                      ffn_hidden=256, seq_len=64, attn_type=attn_type,
                      n_kv_head=2, mtp=mtp)
    torch.manual_seed(0)
    model = GPT(cfg)
    B, T = 3, 32
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    tgt = torch.randint(0, cfg.vocab_size, (B, mtp, T))

    loss, main = model(idx, tgt)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # generation
    out = model.generate(idx[:, :5], max_new_tokens=8, top_k=10)

    # every parameter that requires grad should have received a gradient
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"no grad for: {missing}"
    assert out.shape == (B, 13), out.shape
    print(f"[{attn_type} mtp={mtp}] params={model.num_params()/1e6:.2f}M "
          f"total_loss={loss.item():.3f} main_loss={main.item():.3f} "
          f"grad_norm={gnorm:.2f} gen_ok={tuple(out.shape)}")


if __name__ == "__main__":
    run("diff", mtp=2)
    run("diff", mtp=1)
    run("gqa", mtp=2)
    run("gqa", mtp=1)
    print("ALL SMOKE TESTS PASSED")
