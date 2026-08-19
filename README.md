# koGPT — a Korean LM trained from scratch on Vessl (4× RTX 3090)

A small-to-mid LLM built **from scratch** (own architecture + training loop, no
HuggingFace model import) and trained on the Yonsei Vessl cluster. Korean-first,
with a one-line switch to an English fallback.

## Project roadmap (burn the GPU hours in order)
One codebase, switched by env vars:
1. **Differential Transformer + MTP** — `KOGPT_ATTN=diff KOGPT_MTP=2` *(default)*
2. **GPT-3/Llama/Qwen-style dense** — `KOGPT_ATTN=gqa KOGPT_MTP=1`
3. **Mamba-2 hybrid** — future (separate block, not in this file yet)

## Architecture (`KOGPT_ATTN`)
- **`diff` — Differential Attention** (Ye et al., 2024): two softmax attention
  maps subtracted with a learned, depth-scheduled λ, cancelling attention noise.
  Param/FLOP-matched to normal attention (projections stay d→d).
- **`gqa` — Llama/Qwen2-style**: RoPE + grouped-query attention.
- Both share **RoPE + RMSNorm (pre-norm) + SwiGLU + QK-Norm + tied embeddings**,
  and fused SDPA (FlashAttention-2 on Ampere).
- **Multi-Token Prediction** (`KOGPT_MTP`): independent heads predict t+1..t+MTP
  (Gloeckle et al., 2024). Head 0 is the tied lm_head; extras add a small aux
  loss and double as speculative-decoding heads.

## What's Korean-favorable here
The biggest lever is the **tokenizer**, not the architecture:
- We train our own **SentencePiece BPE (vocab 48k, `byte_fallback`)** on Korean
  text, so Hangul is segmented into subwords instead of ~3 bytes/char. This
  roughly halves tokens/sentence → ~2× effective data & context per FLOP.
- **NFC** normalization + an inline **Hangul-ratio filter** (no offline cleaning).

## Data mixture (`config.py` → `_DATA`)
A weighted **streaming** mixture (no local downloads, no offline cleaning):
| source | weight | what |
|--------|--------|------|
| `HuggingFaceFW/fineweb-2` `kor_Hang` | 0.93 | modern Korean web text, cleaned/deduped by HF |
| `wikimedia/wikisource` `20231201.ko` | 0.07 | **public-domain** Korean literature (근대 소설/시/고전) for literary 문체 |

The literature stream is small, so it **cycles** (`interleave_datasets`,
`all_exhausted`) to actually reach its 7% share and imprint narrative style.

> **On web novels:** modern Korean web novels (판타지/로맨스 등) are copyrighted
> commercial works with no clean/licensed dataset — scraping them would infringe,
> so they're intentionally **not** included. Public-domain classic literature
> (Wikisource) is the licensable way to nudge toward a "novel" 문체. If you have
> **AI Hub 문학/도서** access, add it as another `_src(...)` for a bigger literary mix.

## Model sizes (`KOGPT_SIZE`), diff variant, verified param counts
| size | non-emb params | ~tokens | ~wall-clock on 4×3090 |
|------|--------|---------|-----------------------|
| `warmup` | ~122M | 2.5B | half a day |
| `medium` (default) | ~357M | 7B | 2–3 days |
| `large`  | ~775M | 15B | 5–7 days |

Total compute ≈ `6 × params × tokens` — dial `KOGPT_SIZE` / `target_tokens`
(in `config.py`) to burn exactly the GPU-hours you have left.

## Vessl constraints handled
- **50GiB disk & 24h Workspace cap don't change** → train as a **Run** (120h),
  checkpoints go to a **persistent volume** (`/ckpt`), and a **SIGTERM handler**
  saves + exits cleanly so relaunching **auto-resumes**.
- **No local downloads** → all data is streamed inside the Vessl environment.
- **Disk/inode exhaustion** → output is exactly **two flat `.bin` files** (no
  shard directories), plus a `guard.py` that watches free space *and inodes*,
  purges the HF cache, and rotates checkpoints (keep last 2 + best).

## How to run
Org `YS-SUMMER`, cluster `yonsei-ai-gpu`, storage `vessl-storage` (already exist).

1. Push this repo to GitHub and set the `url:` in `vessl/*.yaml` (manual: Github 연동).
2. Create two persistent volumes (safe, no GPU):
   ```bash
   vessl storage create-volume kogpt-data --storage-name vessl-storage
   vessl storage create-volume kogpt-ckpt --storage-name vessl-storage
   ```
3. From your Mac (**VPN on!**):
   ```bash
   vessl run create -f vessl/run_prepare.yaml -w   # step 1: tokenizer + .bin (gpu-1)
   vessl run create -f vessl/run_train.yaml   -w   # step 2: train on gpu-4
   ```
   (CLI lives at `~/.local/share/vessl-cli-venv/bin/vessl`; already logged in.)
4. Watch it: `vessl run list` / `vessl run logs <name>`.
5. When done:
   ```bash
   python eval.py   --ckpt /ckpt/best.pt              # perplexity
   python sample.py --ckpt /ckpt/best.pt --prompt "인공지능은"
   ```

## English fallback
If the first eval looks weak or Korean data is slow, flip **one env var**
(`KOGPT_LANG: en`) in both YAMLs — it switches to FineWeb-Edu and an English
tokenizer/data pipeline with no code changes.

## Files
- `config.py` — all knobs (language, sizes, optimizer, paths).
- `model.py` — the GPT (from scratch).
- `tokenizer/train_tokenizer.py` — SentencePiece training from the stream.
- `data/prepare.py` — stream → filter → tokenize → 2 `.bin` files.
- `data/filters.py` — inline Hangul/length filters.
- `train.py` — DDP + bf16 + compile + resumable checkpointing.
- `guard.py` — disk/inode guard + checkpoint rotation.
- `sample.py`, `eval.py` — generation and perplexity.
- `vessl/*.yaml` — the two Runs.
