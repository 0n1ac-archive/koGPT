"""Train a Korean-favorable SentencePiece BPE tokenizer from the data stream.

Why our own tokenizer (the single biggest Korean lever):
- GPT-2 byte-level BPE spends ~3 bytes/char on Hangul -> high token fertility.
- A SP-BPE trained on Korean text segments Hangul into subwords/morphemes,
  roughly halving tokens per sentence -> ~2x effective data & context per FLOP.
- byte_fallback=True keeps it OOV-free for rare chars/emoji/foreign words.

We stream ~2GB of already-cleaned FineWeb-2 text to ONE temp file (inode-safe),
train SP on it (minutes on CPU), then delete the temp file.

Run inside the Vessl environment:
    python tokenizer/train_tokenizer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig, TokenizerConfig  # noqa: E402
from data.sources import iter_documents          # noqa: E402


def main():
    dc = DataConfig()
    tc = TokenizerConfig()
    os.makedirs(os.path.dirname(tc.model_prefix), exist_ok=True)
    os.makedirs(dc.hf_cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", dc.hf_cache)

    import sentencepiece as spm

    sample_path = tc.model_prefix + ".sample.txt"   # single file -> inode-safe
    srcs = ", ".join(f"{s['hf_path']}:{s['hf_name']}" for s in dc.sources)
    print(f"[tok] streaming up to {tc.sample_bytes/1e9:.1f}GB from mixture "
          f"[{srcs}] for tokenizer training", flush=True)

    written = 0
    with open(sample_path, "w", encoding="utf-8") as f:
        for text in iter_documents(dc):            # same weighted mixture as prepare.py
            f.write(text.replace("\n", " ") + "\n")
            written += len(text.encode("utf-8"))
            if written >= tc.sample_bytes:
                break
    print(f"[tok] wrote {written/1e9:.2f}GB sample -> training SentencePiece", flush=True)

    spm.SentencePieceTrainer.train(
        input=sample_path,
        model_prefix=tc.model_prefix,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
        character_coverage=tc.character_coverage,
        byte_fallback=tc.byte_fallback,
        # special ids: 0=unk 1=bos 2=eos 3=pad
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        num_threads=os.cpu_count() or 8,
        train_extremely_large_corpus=True,
        input_sentence_size=5_000_000,     # cap sentences for speed
        shuffle_input_sentence=True,
    )
    os.remove(sample_path)                 # reclaim the ~2GB immediately
    print(f"[tok] done -> {tc.model_file}", flush=True)


if __name__ == "__main__":
    main()
