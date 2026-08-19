"""A decoder LM built from scratch, with two selectable attention types and
multi-token prediction (MTP). No HuggingFace model import.

attn_type="diff"  -> Differential Attention (Ye et al., 2024): two softmax maps
                     subtracted with a learned, depth-scheduled lambda. Cancels
                     attention noise. Param/FLOP-matched to standard attention
                     (all projections stay d->d), per the paper.
attn_type="gqa"   -> a modern Llama/Qwen2-style block (RoPE + GQA + QK-norm),
                     for the "GPT-3/Llama/Qwen" project.

Both share: RoPE, RMSNorm (pre-norm), SwiGLU MLP, tied input/output embeddings,
QK-Norm, and torch's fused SDPA (FlashAttention-2 on Ampere / RTX 3090).

MTP: `mtp` independent linear heads read the final hidden state and predict
tokens t+1 .. t+mtp (Gloeckle et al., 2024 style). Head 0 is the tied lm_head;
extra heads add a small auxiliary loss and are reusable for speculative decoding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]   # (1,1,T,hd/2)


def apply_rope(x, cos, sin):
    # x: (B, nH, T, hd)
    T = x.shape[2]
    c, s = cos[:, :, :T, :], sin[:, :, :T, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * c - x2 * s
    rx2 = x1 * s + x2 * c
    return torch.stack((rx1, rx2), dim=-1).flatten(-2).type_as(x)


def _rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def lambda_init_fn(layer_idx):
    # depth schedule from the Differential Transformer paper
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class DiffAttention(nn.Module):
    """Differential Attention, param-matched: all projections are d->d.

    We use h "value heads" of dim 2*D and 2h "score heads" of dim D, where
    D = head_dim and h = d / (2D). Q1/K1 and Q2/K2 form the two attention maps.
    """
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d, D = cfg.n_embd, cfg.head_dim
        assert d % (2 * D) == 0, "n_embd must be divisible by 2*head_dim"
        self.h = d // (2 * D)          # value heads
        self.D = D
        self.qk_norm = cfg.qk_norm
        self.dropout = cfg.dropout

        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.subnorm = RMSNorm(2 * D)  # per-head norm on the differential output

        self.lambda_init = lambda_init_fn(layer_idx)
        # learnable lambda reparam (per score-head dim D)
        self.lq1 = nn.Parameter(torch.randn(D) * 0.1)
        self.lk1 = nn.Parameter(torch.randn(D) * 0.1)
        self.lq2 = nn.Parameter(torch.randn(D) * 0.1)
        self.lk2 = nn.Parameter(torch.randn(D) * 0.1)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        h, D = self.h, self.D
        # 2h score-heads for Q,K ; h value-heads of dim 2D for V
        q = self.wq(x).view(B, T, 2 * h, D).transpose(1, 2)   # (B,2h,T,D)
        k = self.wk(x).view(B, T, 2 * h, D).transpose(1, 2)
        v = self.wv(x).view(B, T, h, 2 * D).transpose(1, 2)    # (B,h,T,2D)
        if self.qk_norm:
            q, k = _rms(q), _rms(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        q1, q2 = q[:, :h], q[:, h:]        # (B,h,T,D)
        k1, k2 = k[:, :h], k[:, h:]
        a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True,
                                            dropout_p=self.dropout if self.training else 0.0)
        a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True,
                                            dropout_p=self.dropout if self.training else 0.0)
        lam = (torch.exp((self.lq1 * self.lk1).sum())
               - torch.exp((self.lq2 * self.lk2).sum()) + self.lambda_init)
        out = a1 - lam * a2                              # (B,h,T,2D)
        out = self.subnorm(out) * (1 - self.lambda_init)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.wo(out)


class GQAttention(nn.Module):
    """Standard modern attention: RoPE + grouped-query + QK-norm (Llama/Qwen2)."""
    def __init__(self, cfg, layer_idx):
        super().__init__()
        d, D = cfg.n_embd, cfg.head_dim
        assert d % D == 0
        self.nh = d // D
        self.nkv = cfg.n_kv_head
        assert self.nh % self.nkv == 0
        self.D = D
        self.qk_norm = cfg.qk_norm
        self.dropout = cfg.dropout
        self.wq = nn.Linear(d, self.nh * D, bias=False)
        self.wk = nn.Linear(d, self.nkv * D, bias=False)
        self.wv = nn.Linear(d, self.nkv * D, bias=False)
        self.wo = nn.Linear(self.nh * D, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        q = self.wq(x).view(B, T, self.nh, self.D).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.D).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.D).transpose(1, 2)
        if self.qk_norm:
            q, k = _rms(q), _rms(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, d)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = (DiffAttention if cfg.attn_type == "diff" else GQAttention)(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)

        # multi-token prediction heads: head 0 tied to embeddings, rest are new
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) for _ in range(cfg.mtp)])
        if cfg.tie_embeddings:
            self.heads[0].weight = self.tok_emb.weight
        self.register_buffer(
            "mtp_weights",
            torch.tensor([1.0] + [cfg.mtp_weight] * (cfg.mtp - 1)), persistent=False)

        cos, sin = build_rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def trunk(self, idx):
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        return self.norm_f(x)

    def forward(self, idx, targets=None):
        """targets: None (inference) or (B, K, T) with K<=mtp future-token streams.
        Returns (logits, None) at inference, else (loss_total, loss_main)."""
        if targets is None:
            h = self.trunk(idx)
            return self.heads[0](h[:, [-1], :]), None
        h = self.trunk(idx)
        K = targets.shape[1]
        loss_total, loss_main = 0.0, None
        for k in range(K):
            logits = self.heads[k](h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets[:, k].reshape(-1), ignore_index=-1)
            loss_total = loss_total + self.mtp_weights[k] * loss
            if k == 0:
                loss_main = loss.detach()
        return loss_total, loss_main

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)                 # head 0, last position
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
