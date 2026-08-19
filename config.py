"""Single source of truth for the whole project.

Language switch lives here: set LANG = "ko" (default) or "en".
Everything else (data source, tokenizer path, eval prompts) follows from it.
"""
from dataclasses import dataclass, field, asdict
import os


# ----------------------------------------------------------------------------
# Language / data source. Flip LANG to "en" for the FineWeb-Edu fallback.
# ----------------------------------------------------------------------------
LANG = os.environ.get("KOGPT_LANG", "ko")

def _src(hf_path, hf_name, weight, split="train", text_key="text",
         min_hangul_ratio=0.30, min_chars=200):
    return dict(hf_path=hf_path, hf_name=hf_name, split=split, text_key=text_key,
                weight=weight, min_hangul_ratio=min_hangul_ratio, min_chars=min_chars)


# Weighted streaming mixtures. All sources are ALREADY cleaned/deduped or
# public-domain, so we need no separate offline cleaning pass.
_DATA = {
    "ko": [
        # bulk: modern Korean web text (cleaned/deduped by HF)
        _src("HuggingFaceFW/fineweb-2", "kor_Hang", weight=0.93),
        # up-weighted: PUBLIC-DOMAIN Korean literature (근대 소설/시/고전) for
        # narrative/literary 문체. Small corpus, so it cycles to fill its share.
        # min_chars lowered so short poems survive the filter.
        _src("wikimedia/wikisource", "20231201.ko", weight=0.07, min_chars=100),
        # To add AI Hub 문학/도서 later, append another _src(...) here.
    ],
    "en": [
        _src("HuggingFaceFW/fineweb-edu", "sample-10BT", weight=1.0,
             min_hangul_ratio=0.0),   # Hangul filter disabled for English
    ],
}


@dataclass
class DataConfig:
    lang: str = LANG
    # storage paths — BIN_DIR is the 50GiB ephemeral disk (2 files only).
    bin_dir: str = os.environ.get("KOGPT_BIN_DIR", "/data/bin")
    hf_cache: str = os.environ.get("HF_DATASETS_CACHE", "/data/hf_cache")
    target_tokens: int = 7_000_000_000    # ~Chinchilla-optimal for 350M
    val_tokens: int = 10_000_000          # held-out validation
    dtype: str = "uint16"                 # vocab < 65536 -> uint16 is enough
    seq_len: int = 1024

    def __post_init__(self):
        self.sources = _DATA[self.lang]     # list of weighted streaming sources

    @property
    def train_bin(self):
        return os.path.join(self.bin_dir, f"train_{self.lang}.bin")

    @property
    def val_bin(self):
        return os.path.join(self.bin_dir, f"val_{self.lang}.bin")


@dataclass
class TokenizerConfig:
    vocab_size: int = 48000
    model_prefix: str = os.environ.get("KOGPT_TOK", "/data/tok/kogpt_sp")
    sample_bytes: int = 2_000_000_000     # ~2GB sampled from stream to train SP
    character_coverage: float = 0.9995    # high coverage for Hangul
    model_type: str = "bpe"
    byte_fallback: bool = True            # no OOV, robust to rare chars

    @property
    def model_file(self):
        return self.model_prefix + ".model"


# ----------------------------------------------------------------------------
# Model sizes. Pick with KOGPT_SIZE=medium (default) or warmup.
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = "medium"
    vocab_size: int = 48000     # overwritten from tokenizer at train time
    n_layer: int = 24
    n_embd: int = 1024
    head_dim: int = 64          # per-head dim (RoPE dim)
    ffn_hidden: int = 2816      # SwiGLU hidden, multiple of 256
    seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0

    # --- architecture switches (project #1 = diff, project #2 = gqa) ---
    attn_type: str = os.environ.get("KOGPT_ATTN", "diff")   # "diff" | "gqa"
    qk_norm: bool = True
    n_kv_head: int = 4          # gqa path only (KV heads); diff path ignores it

    # --- multi-token prediction ---
    mtp: int = int(os.environ.get("KOGPT_MTP", "2"))        # #future tokens (1 = off)
    mtp_weight: float = 0.3     # loss weight for each auxiliary (t+2, ...) head


_SIZES = {
    # ~124M, quick warmup run (half a day on 4x3090)
    "warmup": dict(name="warmup", n_layer=12, n_embd=768,  head_dim=64, ffn_hidden=2048),
    # ~350M, the main run (2-3 days on 4x3090)
    "medium": dict(name="medium", n_layer=24, n_embd=1024, head_dim=64, ffn_hidden=2816),
    # ~774M, only if you have hours to burn (5-7 days)
    "large":  dict(name="large",  n_layer=36, n_embd=1280, head_dim=64, ffn_hidden=3456),
}


def get_model_config():
    size = os.environ.get("KOGPT_SIZE", "medium")
    cfg = ModelConfig(**_SIZES[size])
    if os.environ.get("KOGPT_ATTN"):
        cfg.attn_type = os.environ["KOGPT_ATTN"]
    return cfg


@dataclass
class TrainConfig:
    # optimization  (conservative micro_bsz for a safe first run on 24GB;
    # raise micro_bsz and lower grad_accum once you confirm memory headroom)
    micro_bsz: int = 4           # sequences per GPU per micro-step
    grad_accum: int = 32         # -> global tokens/step = 4gpu*4*1024*32 ~= 0.5M
    max_steps: int = 14000       # ~7B tokens at 0.5M/step
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # runtime / robustness (constraints #1 and #3)
    ckpt_dir: str = os.environ.get("KOGPT_CKPT_DIR", "/ckpt")   # <- Storage mount
    ckpt_every: int = 500
    eval_every: int = 500
    eval_iters: int = 100
    keep_last: int = 2           # checkpoint rotation to save disk/inodes
    log_every: int = 20

    # disk/inode guard thresholds
    min_free_gb: float = 3.0
    min_free_inodes: int = 200_000

    # precision / speed (Ampere 3090)
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337


def all_configs():
    return dict(
        lang=LANG,
        data=asdict(DataConfig()),
        tokenizer=asdict(TokenizerConfig()),
        model=asdict(get_model_config()),
        train=asdict(TrainConfig()),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(all_configs(), indent=2, ensure_ascii=False))
