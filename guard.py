"""Disk + inode guard (constraint #3: Vessl nodes run out of both, inodes first).

- free_space() reports free GB and free inodes for a path.
- ensure_space() aborts *gracefully* (returns False) before we hit a hard ENOSPC.
- purge_hf_cache() reclaims the biggest inode hog (datasets streaming cache).
"""
import os
import shutil


def free_space(path):
    """Return (free_gb, free_inodes) for the filesystem holding `path`."""
    path = path if os.path.exists(path) else os.path.dirname(path) or "."
    st = os.statvfs(path)
    free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
    free_inodes = st.f_favail            # available inodes (0 on some FS = "unknown")
    return free_gb, free_inodes


def ensure_space(path, min_free_gb, min_free_inodes):
    """True if we still have headroom. Prints a clear reason if not."""
    gb, inodes = free_space(path)
    ok = True
    if gb < min_free_gb:
        print(f"[guard] LOW DISK on {path}: {gb:.1f}GB free < {min_free_gb}GB", flush=True)
        ok = False
    # f_favail == 0 usually means the FS doesn't report inodes; don't false-alarm
    if inodes and inodes < min_free_inodes:
        print(f"[guard] LOW INODES on {path}: {inodes} free < {min_free_inodes}", flush=True)
        ok = False
    return ok


def purge_hf_cache(hf_cache):
    """Delete the HuggingFace datasets streaming cache (millions of tiny files)."""
    for d in (hf_cache, os.path.join(hf_cache, "downloads")):
        if os.path.isdir(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
                print(f"[guard] purged HF cache: {d}", flush=True)
            except Exception as e:  # noqa
                print(f"[guard] failed purging {d}: {e}", flush=True)
    os.makedirs(hf_cache, exist_ok=True)


def rotate_checkpoints(ckpt_dir, keep_last, best_name="best.pt"):
    """Keep only the newest `keep_last` step checkpoints plus best.pt."""
    if not os.path.isdir(ckpt_dir):
        return
    steps = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("step_") and f.endswith(".pt"):
            try:
                steps.append((int(f[len("step_"):-3]), f))
            except ValueError:
                pass
    steps.sort()
    for _, f in steps[:-keep_last] if keep_last > 0 else steps:
        try:
            os.remove(os.path.join(ckpt_dir, f))
            print(f"[guard] rotated out old checkpoint: {f}", flush=True)
        except OSError:
            pass


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "."
    gb, ino = free_space(p)
    print(f"{p}: {gb:.1f} GB free, {ino} inodes free")
