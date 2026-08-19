"""Train the GPT from scratch on 4x RTX 3090 with DDP.

Launch inside a Vessl Run (see vessl/run.yaml):
    torchrun --standalone --nproc_per_node=4 train.py

Robustness (Vessl constraints #1, #3):
- Checkpoints are written to KOGPT_CKPT_DIR, which must be a mounted Storage
  volume (NOT the 50GiB ephemeral disk that resets when the Run stops).
- On SIGTERM (Vessl sends it when the 24h/120h runtime cap is hit) we checkpoint
  and exit cleanly; relaunching the Run auto-resumes from the latest checkpoint.
- Checkpoints are rotated (keep last N + best) and a disk/inode guard runs before
  every save so we never wedge the node.
"""
import os
import sys
import glob
import math
import time
import signal
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DataConfig, TokenizerConfig, TrainConfig, get_model_config  # noqa
from model import GPT                                                          # noqa
import guard                                                                   # noqa

_STOP = False

# Persist checkpoints to a storage volume (storage volumes can't be run-mounted,
# so we copy files with the auto-authed `vessl storage copy-file`). Best-effort:
# a failed upload/download never kills training. Set KOGPT_CKPT_VOLUME to enable.
CKPT_VOLUME = os.environ.get("KOGPT_CKPT_VOLUME")  # e.g. volume://vessl-storage/kogpt-ckpt


def _vol_upload(local_path, as_name=None):
    """Upload a checkpoint to the volume. `vessl storage copy-file` appends the
    SOURCE basename to the DEST directory, so to control the remote name we stage
    the file under that name first (hardlink; no extra disk)."""
    if not CKPT_VOLUME or not os.path.exists(local_path):
        return
    src = local_path
    if as_name and os.path.basename(local_path) != as_name:
        src = os.path.join(os.path.dirname(local_path) or ".", as_name)
        try:
            if os.path.exists(src):
                os.remove(src)
            os.link(local_path, src)
        except OSError:
            import shutil
            shutil.copy2(local_path, src)
    try:
        subprocess.run(["vessl", "storage", "copy-file", src, CKPT_VOLUME],
                       check=True, capture_output=True, timeout=3600)   # DEST = dir
        print(f"[train] uploaded {os.path.basename(src)} to volume", flush=True)
    except Exception as e:  # noqa
        print(f"[train] volume upload failed ({os.path.basename(src)}): {e}", flush=True)


def _vol_download_resume(ckpt_dir):
    """Pull latest.pt from the volume into ckpt_dir/latest.pt. Returns path or None."""
    if not CKPT_VOLUME:
        return None
    try:
        r = subprocess.run(["vessl", "storage", "copy-file",
                            f"{CKPT_VOLUME}/latest.pt", ckpt_dir + "/"],  # DEST = dir
                           capture_output=True, timeout=3600)
        dst = os.path.join(ckpt_dir, "latest.pt")
        if r.returncode == 0 and os.path.exists(dst):
            print("[train] downloaded latest.pt from volume", flush=True)
            return dst
    except Exception as e:  # noqa
        print(f"[train] no volume resume available ({e})", flush=True)
    return None


def _handle_sigterm(signum, frame):
    global _STOP
    _STOP = True
    print("[train] SIGTERM received -> will checkpoint and exit.", flush=True)


def is_ddp():
    return int(os.environ.get("RANK", -1)) != -1


def get_batch(data, seq_len, batch_size, device, mtp=1):
    """Return x:(B,T) and targets:(B,mtp,T) where targets[:,k] is shifted by k+1."""
    ix = torch.randint(len(data) - seq_len - mtp - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    tgt = torch.stack([
        torch.stack([torch.from_numpy(data[i + 1 + k:i + 1 + k + seq_len].astype(np.int64))
                     for k in range(mtp)])
        for i in ix])                                  # (B, mtp, T)
    x = x.pin_memory().to(device, non_blocking=True)
    tgt = tgt.pin_memory().to(device, non_blocking=True)
    return x, tgt


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def find_latest_ckpt(ckpt_dir):
    steps = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not steps:
        return None
    return max(steps, key=lambda p: int(os.path.basename(p)[5:-3]))


def save_ckpt(path, raw_model, optimizer, step, best_val, mcfg, is_master):
    if not is_master:
        return
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_config": mcfg.__dict__,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, tmp)
    os.replace(tmp, path)   # atomic
    print(f"[train] saved {path}", flush=True)


@torch.no_grad()
def evaluate(model, val_data, tcfg, mcfg, device, ctx):
    model.eval()
    losses = torch.zeros(tcfg.eval_iters)
    for i in range(tcfg.eval_iters):
        x, tgt = get_batch(val_data, mcfg.seq_len, tcfg.micro_bsz, device, mtp=1)
        with ctx:
            loss, _ = model(x, tgt)          # main-head loss only (K=1)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    dcfg, tokcfg, tcfg, mcfg = DataConfig(), TokenizerConfig(), TrainConfig(), get_model_config()

    # --- DDP setup ---
    ddp = is_ddp()
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank, local_rank, world = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_master = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    torch.manual_seed(tcfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- tokenizer vocab -> model vocab ---
    if os.path.exists(tokcfg.model_file):
        import sentencepiece as spm
        mcfg.vocab_size = spm.SentencePieceProcessor(model_file=tokcfg.model_file).vocab_size()
    if is_master:
        print(f"[train] lang={dcfg.lang} size={mcfg.name} vocab={mcfg.vocab_size} "
              f"world={world}", flush=True)

    # --- data (memmap, zero extra inodes) ---
    train_data = np.memmap(dcfg.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(dcfg.val_bin, dtype=np.uint16, mode="r")

    # --- model ---
    torch.manual_seed(tcfg.seed)
    model = GPT(mcfg).to(device)
    if is_master:
        print(f"[train] params: {model.num_params()/1e6:.1f}M (non-embedding)", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[tcfg.dtype]
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.startswith("cuda") \
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    optimizer = model.configure_optimizers(
        tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2), "cuda")

    # --- resume (prefer a local checkpoint; else pull latest.pt from volume) ---
    os.makedirs(tcfg.ckpt_dir, exist_ok=True)
    step, best_val = 0, float("inf")
    if is_master and not find_latest_ckpt(tcfg.ckpt_dir):
        _vol_download_resume(tcfg.ckpt_dir)   # single shared FS across ranks (1 node)
    if ddp:
        dist.barrier()
    latest = find_latest_ckpt(tcfg.ckpt_dir)
    if latest is None:
        resume_pt = os.path.join(tcfg.ckpt_dir, "latest.pt")
        latest = resume_pt if os.path.exists(resume_pt) else None
    if latest:
        try:
            ck = torch.load(latest, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            step, best_val = ck["step"], ck.get("best_val", float("inf"))
            if "torch_rng" in ck:
                torch.set_rng_state(ck["torch_rng"].cpu())
            if is_master:
                print(f"[train] resumed from {latest} at step {step}", flush=True)
        except Exception as e:  # noqa  (corrupt/incompatible ckpt -> start fresh)
            step, best_val = 0, float("inf")
            if is_master:
                print(f"[train] could not load {latest} ({e}); starting fresh", flush=True)

    if tcfg.compile:
        try:
            model = torch.compile(model)
        except Exception as e:  # noqa
            print(f"[train] torch.compile failed ({e}); continuing eager.", flush=True)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model

    # --- train loop ---
    t0 = time.time()
    tokens_per_step = tcfg.micro_bsz * mcfg.seq_len * tcfg.grad_accum * world
    model.train()
    while step < tcfg.max_steps and not _STOP:
        lr = lr_at(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(tcfg.grad_accum):
            x, tgt = get_batch(train_data, mcfg.seq_len, tcfg.micro_bsz, device, mtp=mcfg.mtp)
            if ddp:
                model.require_backward_grad_sync = (micro == tcfg.grad_accum - 1)
            with ctx:
                loss, loss_main = model(x, tgt)
                loss = loss / tcfg.grad_accum
            loss.backward()
            loss_accum += loss_main.item() / tcfg.grad_accum   # log the main-head loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if is_master and step % tcfg.log_every == 0:
            dt = time.time() - t0
            tps = tokens_per_step * tcfg.log_every / dt if step else 0
            print(f"[train] step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} "
                  f"| grad {norm:.2f} | {tps/1e3:.0f}k tok/s", flush=True)
            t0 = time.time()

        # eval + checkpoint
        if step > 0 and step % tcfg.eval_every == 0:
            val = evaluate(model, val_data, tcfg, mcfg, device, ctx)
            if is_master:
                print(f"[train] step {step} | val loss {val:.4f} | ppl {math.exp(val):.1f}",
                      flush=True)
                if guard.ensure_space(tcfg.ckpt_dir, tcfg.min_free_gb, tcfg.min_free_inodes):
                    step_path = os.path.join(tcfg.ckpt_dir, f"step_{step}.pt")
                    save_ckpt(step_path, raw_model, optimizer, step, best_val, mcfg, is_master)
                    _vol_upload(step_path, "latest.pt")   # rolling resume point on volume
                    if val < best_val:
                        best_val = val
                        best_path = os.path.join(tcfg.ckpt_dir, "best.pt")
                        save_ckpt(best_path, raw_model, optimizer, step, best_val, mcfg, is_master)
                        _vol_upload(best_path, "best.pt")  # best model, persisted
                    guard.rotate_checkpoints(tcfg.ckpt_dir, tcfg.keep_last)
                else:
                    guard.purge_hf_cache(dcfg.hf_cache)
            if ddp:
                dist.barrier()
        step += 1

    # final save (also runs on SIGTERM path) — upload so a relaunch can resume
    if is_master:
        final_path = os.path.join(tcfg.ckpt_dir, f"step_{step}.pt")
        save_ckpt(final_path, raw_model, optimizer, step, best_val, mcfg, is_master)
        _vol_upload(final_path, "latest.pt")
        print(f"[train] stopped at step {step} (STOP={_STOP})", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
