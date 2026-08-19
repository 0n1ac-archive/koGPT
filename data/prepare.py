"""Stream -> filter -> tokenize -> write TWO .bin files (train + val).

Design choices driven by the Vessl constraints:
- Data is streamed (constraint #2): nothing is downloaded to your laptop, and the
  raw corpus is never materialized on disk — only the tokenized .bin.
- Output is exactly 2 flat uint16 files (constraint #3): no directory of millions
  of small shards, so we never exhaust inodes.
- A disk/inode guard runs periodically and aborts *gracefully* if space runs low,
  so a full .bin (val first, then as much train as fits) is always usable.

Run inside the Vessl environment AFTER the tokenizer exists:
    python data/prepare.py
"""
import os
import sys
import time
import array
import unicodedata
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, TokenizerConfig  # noqa: E402
from data.filters import keep_document           # noqa: E402
import guard                                      # noqa: E402

_SP = None
_EOS = 2


def _init_worker(model_file):
    global _SP
    import sentencepiece as spm
    _SP = spm.SentencePieceProcessor(model_file=model_file)


def _encode(text):
    ids = _SP.encode(text, out_type=int)
    ids.append(_EOS)
    return ids


def _doc_stream(dc):
    from datasets import load_dataset
    ds = load_dataset(dc.hf_path, name=dc.hf_name, split=dc.hf_split, streaming=True)
    for ex in ds:
        text = ex.get(dc.text_key, "")
        if not text:
            continue
        text = unicodedata.normalize("NFC", text)
        if keep_document(text, dc.min_hangul_ratio):
            yield text


def _write_split(name, out_path, n_tokens, doc_iter, pool, dc, tc):
    """Tokenize from doc_iter until n_tokens written to out_path. Returns written."""
    print(f"[prep] writing {name}: target {n_tokens/1e9:.2f}B tokens -> {out_path}",
          flush=True)
    written = 0
    last_guard = time.time()
    buf = array.array("H")  # unsigned short = uint16
    with open(out_path, "wb") as f:
        for ids in pool.imap(_encode, doc_iter, chunksize=64):
            buf.extend(ids)
            if len(buf) >= 1_000_000:
                np.frombuffer(buf, dtype=np.uint16).tofile(f)
                written += len(buf)
                buf = array.array("H")
                if written % 20_000_000 < 1_000_000:
                    print(f"[prep]   {name}: {written/1e6:.0f}M tokens", flush=True)
            if written >= n_tokens:
                break
            # periodic disk/inode guard
            if time.time() - last_guard > 30:
                last_guard = time.time()
                if not guard.ensure_space(dc.bin_dir, tc_min_gb, tc_min_inodes):
                    print(f"[prep] aborting {name} early to stay within disk/inodes",
                          flush=True)
                    break
        if len(buf):
            np.frombuffer(buf, dtype=np.uint16).tofile(f)
            written += len(buf)
    print(f"[prep] {name} done: {written/1e6:.1f}M tokens ({written*2/1e9:.2f}GB)",
          flush=True)
    return written


# guard thresholds (kept loose here; training uses TrainConfig's tighter ones)
tc_min_gb = 3.0
tc_min_inodes = 200_000


def main():
    dc = DataConfig()
    tc = TokenizerConfig()
    os.makedirs(dc.bin_dir, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", dc.hf_cache)

    if not os.path.exists(tc.model_file):
        sys.exit(f"[prep] tokenizer not found at {tc.model_file}; "
                 f"run tokenizer/train_tokenizer.py first")

    if os.path.exists(dc.train_bin) and os.path.exists(dc.val_bin):
        print(f"[prep] {dc.train_bin} and {dc.val_bin} already exist — skipping.",
              flush=True)
        return

    docs = _doc_stream(dc)
    with mp.Pool(processes=max(2, (os.cpu_count() or 4) - 1),
                 initializer=_init_worker, initargs=(tc.model_file,)) as pool:
        # val first so it always exists even if train is cut short
        _write_split("val", dc.val_bin, dc.val_tokens, docs, pool, dc, tc)
        _write_split("train", dc.train_bin, dc.target_tokens, docs, pool, dc, tc)

    guard.purge_hf_cache(dc.hf_cache)
    print("[prep] all done.", flush=True)


if __name__ == "__main__":
    main()
