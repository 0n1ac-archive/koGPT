"""Weighted streaming mixture of corpora — shared by the tokenizer trainer and
the .bin builder so both see the exact same document distribution.

For Korean we blend a large modern web corpus with a small, up-weighted stream
of PUBLIC-DOMAIN literature (Korean Wikisource: 근대 소설/시/고전) so the model
picks up some literary/narrative 문체 without touching copyrighted web novels.
"""
import os
import unicodedata

# hf_xet's background threads can crash at interpreter shutdown
# (PyGILState_Release / core dump). Force the classic HTTP downloader instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data.filters import keep_document


def _normalize_map(text_key, mhr, min_chars):
    def fn(ex):
        return {"text": ex.get(text_key) or "", "_mhr": mhr, "_minc": min_chars}
    return fn


def build_mixture(dc, seed=42):
    """Return a streaming HF dataset yielding uniform {'text','_mhr','_minc'}."""
    from datasets import load_dataset, interleave_datasets

    dss, probs = [], []
    for src in dc.sources:
        ds = load_dataset(src["hf_path"], name=src["hf_name"],
                          split=src["split"], streaming=True)
        ds = ds.map(_normalize_map(src["text_key"], src["min_hangul_ratio"],
                                   src["min_chars"]))
        # force a uniform 3-column schema so interleave_datasets won't choke on
        # heterogeneous source features (robust across datasets streaming versions)
        ds = ds.select_columns(["text", "_mhr", "_minc"])
        dss.append(ds)
        probs.append(src["weight"])
    if len(dss) == 1:
        return dss[0]
    s = sum(probs)
    probs = [p / s for p in probs]
    # all_exhausted -> the small literature stream cycles (repeats) while the
    # big web stream keeps flowing, so the mix ratio holds for the whole run.
    return interleave_datasets(dss, probabilities=probs, seed=seed,
                               stopping_strategy="all_exhausted")


def iter_documents(dc, seed=42):
    """Yield NFC-normalized, filtered text strings from the weighted mixture."""
    for ex in build_mixture(dc, seed=seed):
        text = ex["text"]
        if not text:
            continue
        text = unicodedata.normalize("NFC", text)
        if keep_document(text, ex["_mhr"], min_chars=ex["_minc"]):
            yield text
