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
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, TokenizerConfig  # noqa: E402
from data.sources import iter_documents          # noqa: E402
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


# guard thresholds (kept loose here; training uses TrainConfig's tighter ones)
tc_min_gb = 3.0
tc_min_inodes = 200_000


def _write_all(splits, docs, pool, dc):
    """Write every split in ONE continuous imap pass over `docs`.

    We deliberately consume a single generator through a single imap call — the
    val split is filled first, then we roll straight into train WITHOUT breaking
    and restarting imap (a second imap over the same streaming generator would
    deadlock against the first call's still-running feeder thread).
    """
    idx = 0
    name, path, n_tokens = splits[idx]
    print(f"[prep] writing {name}: target {n_tokens/1e6:.1f}M tokens -> {path}", flush=True)
    f = open(path, "wb")
    written, last_print, last_guard = 0, 0, time.time()
    buf = array.array("H")  # uint16

    def flush():
        nonlocal written
        if buf:
            np.frombuffer(buf, dtype=np.uint16).tofile(f)
            written += len(buf)
            del buf[:]

    for ids in pool.imap(_encode, docs, chunksize=64):
        buf.extend(ids)
        if len(buf) >= 1_000_000:
            flush()
            if written - last_print >= 5_000_000:
                print(f"[prep]   {name}: {written/1e6:.0f}M tokens", flush=True)
                last_print = written
        # advance to the next split(s) once the current target is met
        while written >= n_tokens:
            f.close()
            print(f"[prep] {name} done: {written/1e6:.1f}M tokens "
                  f"({written*2/1e9:.2f}GB)", flush=True)
            idx += 1
            if idx >= len(splits):
                return
            name, path, n_tokens = splits[idx]
            print(f"[prep] writing {name}: target {n_tokens/1e6:.1f}M tokens -> {path}",
                  flush=True)
            f = open(path, "wb")
            written, last_print = 0, 0
        if time.time() - last_guard > 30:
            last_guard = time.time()
            if not guard.ensure_space(dc.bin_dir, tc_min_gb, tc_min_inodes):
                print("[prep] low disk/inodes — stopping early (files usable)", flush=True)
                flush()
                f.close()
                return
    flush()
    f.close()


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

    docs = iter_documents(dc)
    # val first so it always exists even if train is cut short
    splits = [("val", dc.val_bin, dc.val_tokens),
              ("train", dc.train_bin, dc.target_tokens)]
    with mp.Pool(processes=max(2, (os.cpu_count() or 4) - 1),
                 initializer=_init_worker, initargs=(tc.model_file,)) as pool:
        _write_all(splits, docs, pool, dc)

    guard.purge_hf_cache(dc.hf_cache)
    print("[prep] all done.", flush=True)


if __name__ == "__main__":
    main()
