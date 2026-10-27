import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import CHECKPOINTS, CONFIG, RUNS, SEED, SPLITS, device, hub_kwargs, seed_everything, sha256_of
from train import RunLog, collate, load_model_input, make_batches

hf_logging.set_verbosity_error()


@torch.no_grad()
def embed(model, encoded, lengths, batch_size, pad_id, dev):
    model.eval()
    vectors = np.zeros((len(encoded), model.config.hidden_size), dtype=np.float32)
    for batch in make_batches(lengths, np.arange(len(encoded)), batch_size, bucket=10 ** 9):
        ids, mask = collate(encoded, batch, pad_id, dev)
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        vectors[batch] = ((hidden * weights).sum(dim=1) / weights.sum(dim=1)).cpu().numpy()
    return vectors


def main():
    spec = CONFIG["alternative"]
    run_dir = RUNS / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = CHECKPOINTS / spec["name"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_dir / "train.log")
    seed_everything(SEED)
    dev = device()

    frame = load_model_input(spec["variant"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"], **hub_kwargs())
    encoded = tokenizer(list(frame["code"]), truncation=True, max_length=CONFIG["model"]["max_length"])["input_ids"]
    lengths = np.array([len(x) for x in encoded])
    parts = {split: np.flatnonzero(frame["split"].to_numpy() == split) for split in SPLITS}
    y = frame["label"].to_numpy().astype(int)
    log(f"run {spec['name']}: {spec['role']}, frozen CodeBERT ({spec['pooling']} pooling) + {spec['classifier']}")
    for split in SPLITS:
        log(f"{split:<5} {len(parts[split]):>6} functions, {int(y[parts[split]].sum()):>4} vulnerable")

    model = AutoModel.from_pretrained(CONFIG["model"]["name"], use_safetensors=False, **hub_kwargs()).to(dev)
    started = time.time()
    vectors = embed(model, encoded, lengths, CONFIG["training"]["eval_batch_size"], tokenizer.pad_token_id, dev)
    embed_seconds = time.time() - started
    log(f"embeddings {vectors.shape} computed in {embed_seconds / 60:.1f} min on {dev}")

    started = time.time()
    scaler = StandardScaler().fit(vectors[parts["train"]])
    scaled = scaler.transform(vectors)
    grid = []
    for c in spec["c_grid"]:
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        classifier.fit(scaled[parts["train"]], y[parts["train"]])
        auc = roc_auc_score(y[parts["valid"]], classifier.predict_proba(scaled[parts["valid"]])[:, 1])
        grid.append({"C": c, "valid_roc_auc": float(auc)})
        log(f"C = {c:<6} valid ROC AUC {auc:.4f}")
    best = max(grid, key=lambda row: row["valid_roc_auc"])
    classifier = LogisticRegression(C=best["C"], class_weight="balanced", max_iter=5000)
    classifier.fit(scaled[parts["train"]], y[parts["train"]])
    fit_seconds = time.time() - started
    with open(run_dir / "train_log.csv", "w") as fh:
        fh.write("C,valid_roc_auc\n" + "".join(f"{row['C']},{row['valid_roc_auc']:.6f}\n" for row in grid))

    probabilities = classifier.predict_proba(scaled)[:, 1]
    index = np.concatenate([parts["valid"], parts["test"]])
    predictions = frame.loc[index, ["function_id", "split", "label", "partition"]].assign(probability=probabilities[index])
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    artifact = artifact_dir / "classifier.npz"
    np.savez(artifact, coef=classifier.coef_, intercept=classifier.intercept_, mean=scaler.mean_, scale=scaler.scale_)

    summary = {
        "run": spec["name"],
        "role": spec["role"],
        "method": "frozen CodeBERT embeddings + class-weighted logistic regression",
        "variant": spec["variant"],
        "loss": "class-weighted log loss",
        "class_weights_rule": "balanced",
        "class_weights": {"not_vulnerable": round(float(len(y[parts["train"]]) / (2 * (y[parts["train"]] == 0).sum())), 4),
                          "vulnerable": round(float(len(y[parts["train"]]) / (2 * (y[parts["train"]] == 1).sum())), 4)},
        "model": CONFIG["model"],
        "hyperparameters": {"pooling": spec["pooling"], "C_grid": spec["c_grid"], "C": best["C"], "seed": SEED},
        "rows": {split: int(len(parts[split])) for split in SPLITS},
        "best_epoch": None,
        "best_valid_roc_auc": round(best["valid_roc_auc"], 4),
        "train_seconds": round(embed_seconds + fit_seconds, 1),
        "embed_seconds": round(embed_seconds, 1),
        "fit_seconds": round(fit_seconds, 1),
        "trainable_parameters": int(classifier.coef_.size + classifier.intercept_.size),
        "device": str(dev),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": version("transformers"), "scikit-learn": version("scikit-learn")},
        "checkpoint": str(artifact.relative_to(CHECKPOINTS.parent)),
        "checkpoint_sha256": sha256_of(artifact),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"best C {best['C']} (valid ROC AUC {best['valid_roc_auc']:.4f}); "
        f"total {summary['train_seconds'] / 60:.1f} min")
    return summary


if __name__ == "__main__":
    main()
