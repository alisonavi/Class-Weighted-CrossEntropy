import hashlib
import json
import os
import random
import re
import sqlite3
from pathlib import Path

import numpy as np

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw"
WAREHOUSE = DATA / "warehouse.db"
RUNS = BASE / "runs"
OUT = BASE / "out"
CHECKPOINTS = BASE / "checkpoints"
CONFIG = json.loads((BASE / "config.json").read_text())

SPLITS = ["train", "valid", "test"]
VARIANTS = ["with_p3", "without_p3"]
SEED = CONFIG["seed"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
LIGHT_BLUE = "#86b6ef"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#a9a8a0"
LIGHT_GRAY = "#d4d3cc"

LOG = []


def say(text=""):
    print(text)
    LOG.append(str(text))


def title(text):
    say()
    say("=" * 78)
    say(text)
    say("=" * 78)


def save_log(path):
    Path(path).write_text("\n".join(LOG) + "\n")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(code):
    return hashlib.sha1(re.sub(r"\s+", " ", code).strip().encode("utf-8")).hexdigest()


def connect(read_only=False):
    if read_only:
        con = sqlite3.connect(f"file:{WAREHOUSE}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(WAREHOUSE)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def hub_kwargs():
    from huggingface_hub import try_to_load_from_cache
    spec = CONFIG["model"]
    cached = all(isinstance(try_to_load_from_cache(spec["name"], name, revision=spec["revision"]), str)
                 for name in ["config.json", "vocab.json", "merges.txt", "pytorch_model.bin"])
    return {"revision": spec["revision"], "local_files_only": cached}


def device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chart_style():
    return {"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
            "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED,
            "ytick.color": INK_2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 10,
            "legend.frameon": False, "legend.fontsize": 8}


def tidy_axes(ax, grid="y"):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
