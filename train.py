import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging

from common import CHECKPOINTS, CONFIG, RUNS, SEED, SPLITS, connect, device, hub_kwargs, seed_everything, sha256_of

hf_logging.set_verbosity_error()


class RunLog:
    def __init__(self, path):
        self.path = path
        path.write_text("")

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as fh:
            fh.write(str(text) + "\n")


def load_model_input(variant):
    con = connect(read_only=True)
    frame = pd.read_sql("SELECT function_id, split, code, label, partition FROM model_input "
                        "WHERE variant = ? ORDER BY function_id", con, params=[variant])
    con.close()
    return frame


def class_weights(labels, rule):
    counts = np.bincount(labels, minlength=2)
    balanced = len(labels) / (2 * counts)
    return {"none": np.ones(2), "balanced": balanced, "sqrt": np.sqrt(balanced)}[rule]


def make_batches(lengths, index, batch_size, bucket, rng=None):
    index = np.asarray(index)
    if rng is not None:
        index = rng.permutation(index)
    step = batch_size * bucket
    batches = []
    for start in range(0, len(index), step):
        chunk = index[start:start + step]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches.extend(chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size))
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def collate(encoded, batch, pad_id, dev):
    width = max(len(encoded[i]) for i in batch)
    ids = np.full((len(batch), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(batch), width), dtype=np.int64)
    for row, i in enumerate(batch):
        ids[row, :len(encoded[i])] = encoded[i]
        mask[row, :len(encoded[i])] = 1
    return torch.from_numpy(ids).to(dev), torch.from_numpy(mask).to(dev)


@torch.no_grad()
def predict(model, encoded, lengths, index, batch_size, pad_id, dev):
    model.eval()
    probabilities = np.zeros(len(encoded))
    for batch in make_batches(lengths, index, batch_size, bucket=10 ** 9):
        ids, mask = collate(encoded, batch, pad_id, dev)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        probabilities[batch] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probabilities


def load_base_model(dev):
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model"]["name"], num_labels=2,
                                                               use_safetensors=False, **hub_kwargs())
    return model.to(dev)


def train(run_name, limit=None, epochs=None):
    spec = CONFIG["runs"][run_name]
    hp = CONFIG["training"]
    epochs = epochs or spec["epochs"]
    folder_name = run_name + ("_smoke" if limit else "")
    run_dir = RUNS / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / folder_name
    log = RunLog(run_dir / "train.log")
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    dev = device()

    frame = load_model_input(spec["variant"])
    if limit:
        frame = frame.sample(frac=1, random_state=SEED)
        frame = frame[frame.groupby(["split", "label"]).cumcount() < limit]
        frame = frame.sort_values("function_id").reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"], **hub_kwargs())
    encoded = tokenizer(list(frame["code"]), truncation=True, max_length=CONFIG["model"]["max_length"])["input_ids"]
    lengths = np.array([len(x) for x in encoded])
    parts = {split: np.flatnonzero(frame["split"].to_numpy() == split) for split in SPLITS}
    y = frame["label"].to_numpy().astype(int)
    weights = class_weights(y[parts["train"]], spec["class_weights"])

    log(f"run {folder_name}: {spec['role']}")
    log(f"variant {spec['variant']}, device {dev}, epochs {epochs}, batch {hp['batch_size']}, lr {hp['learning_rate']}")
    for split in SPLITS:
        log(f"{split:<5} {len(parts[split]):>6} functions, {int(y[parts[split]].sum()):>4} vulnerable")
    log(f"class weights ({spec['class_weights']}): not vulnerable {weights[0]:.3f}, vulnerable {weights[1]:.3f}")
    log(f"inputs cut at {CONFIG['model']['max_length']} tokens: {100 * np.mean(lengths >= CONFIG['model']['max_length']):.1f}%")

    model = load_base_model(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": hp["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=hp["learning_rate"])
    steps_per_epoch = len(make_batches(lengths, parts["train"], hp["batch_size"], hp["bucket_size"]))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp["warmup_ratio"] * total_steps), total_steps)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev))
    log(f"trainable parameters {trainable:,}; {steps_per_epoch} steps per epoch, {total_steps} in total")

    history = []
    best = {"epoch": 0, "valid_roc_auc": -1.0}
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        losses = []
        batches = make_batches(lengths, parts["train"], hp["batch_size"], hp["bucket_size"], rng)
        for step, batch in enumerate(batches, start=1):
            ids, mask = collate(encoded, batch, tokenizer.pad_token_id, dev)
            target = torch.from_numpy(y[batch]).to(dev)
            loss = criterion(model(input_ids=ids, attention_mask=mask).logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if step % 100 == 0 or step == len(batches):
                log(f"  epoch {epoch} step {step}/{len(batches)}  loss {np.mean(losses[-100:]):.4f}  "
                    f"{time.time() - epoch_start:.0f}s")
        train_seconds = time.time() - epoch_start
        p_valid = predict(model, encoded, lengths, parts["valid"], hp["eval_batch_size"], tokenizer.pad_token_id, dev)
        y_valid = y[parts["valid"]]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "valid_log_loss": float(log_loss(y_valid, p_valid[parts["valid"]], labels=[0, 1])),
               "valid_roc_auc": float(roc_auc_score(y_valid, p_valid[parts["valid"]])),
               "valid_pr_auc": float(average_precision_score(y_valid, p_valid[parts["valid"]])),
               "learning_rate_end": scheduler.get_last_lr()[0], "train_seconds": round(train_seconds, 1),
               "epoch_seconds": round(time.time() - epoch_start, 1)}
        history.append(row)
        improved = row["valid_roc_auc"] > best["valid_roc_auc"]
        if improved:
            best = {"epoch": epoch, "valid_roc_auc": row["valid_roc_auc"]}
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        log(f"epoch {epoch}: train loss {row['train_loss']:.4f}, valid ROC AUC {row['valid_roc_auc']:.4f}, "
            f"valid PR AUC {row['valid_pr_auc']:.4f}, {row['epoch_seconds']:.0f}s"
            + ("  -> best so far, checkpoint saved" if improved else ""))
        pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
    total_seconds = time.time() - started

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(dev)
    index = np.concatenate([parts["valid"], parts["test"]])
    probabilities = predict(model, encoded, lengths, index, hp["eval_batch_size"], tokenizer.pad_token_id, dev)
    predictions = frame.loc[index, ["function_id", "split", "label", "partition"]].assign(probability=probabilities[index])
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    summary = {
        "run": folder_name,
        "role": spec["role"],
        "method": "CodeBERT fine-tuned end to end with a classification head",
        "variant": spec["variant"],
        "loss": "cross-entropy" if spec["class_weights"] == "none" else "class-weighted cross-entropy",
        "class_weights_rule": spec["class_weights"],
        "class_weights": {"not_vulnerable": round(float(weights[0]), 4), "vulnerable": round(float(weights[1]), 4)},
        "model": CONFIG["model"],
        "hyperparameters": {**hp, "epochs": epochs, "seed": SEED, "limit": limit},
        "rows": {split: int(len(parts[split])) for split in SPLITS},
        "best_epoch": best["epoch"],
        "best_valid_roc_auc": round(best["valid_roc_auc"], 4),
        "train_seconds": round(total_seconds, 1),
        "trainable_parameters": int(trainable),
        "device": str(dev),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": version("transformers"), "scikit-learn": version("scikit-learn")},
        "checkpoint": str(checkpoint.relative_to(CHECKPOINTS.parent)),
        "checkpoint_sha256": sha256_of(checkpoint / "model.safetensors"),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"best epoch {best['epoch']} (valid ROC AUC {best['valid_roc_auc']:.4f}); total {total_seconds / 60:.1f} min")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(CONFIG["runs"]))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(args.run, args.limit, args.epochs)
