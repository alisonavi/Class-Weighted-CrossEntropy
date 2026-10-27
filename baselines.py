import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OUT, RUNS, SEED, SPLITS, VARIANTS, save_log, say, title
from train import load_model_input


def write_run(name, variant, method, role, frame, index, probabilities, seconds, parameters, fixed_threshold=None):
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = frame.loc[index, ["function_id", "split", "label", "partition"]].assign(probability=probabilities[index])
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    summary = {"run": name, "role": role, "method": method, "variant": variant, "class_weights_rule": "balanced",
               "train_seconds": round(seconds, 2), "trainable_parameters": parameters, "fixed_threshold": fixed_threshold, "seed": SEED,
               "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    valid = frame.loc[index, "split"].to_numpy() == "valid"
    y = frame.loc[index, "label"].to_numpy()
    auc = roc_auc_score(y[valid], probabilities[index][valid]) if probabilities[index][valid].std() > 0 else 0.5
    say(f"{name:<24} valid ROC AUC {auc:.4f}  ({seconds:.1f}s)")


def main():
    title("REFERENCE BASELINES")
    for variant in VARIANTS:
        frame = load_model_input(variant)
        parts = {split: np.flatnonzero(frame["split"].to_numpy() == split) for split in SPLITS}
        train, index = parts["train"], np.concatenate([parts["valid"], parts["test"]])
        y = frame["label"].to_numpy().astype(int)

        prevalence = float(y[train].mean())
        write_run(f"majority_{variant}", variant, "always predict the majority class (not vulnerable)",
                  "reference: what accuracy looks like with no model", frame, index,
                  np.full(len(frame), prevalence), 0.0, 0, fixed_threshold=0.5)

        started = time.time()
        length = np.log1p(frame["code"].str.len().to_numpy()).reshape(-1, 1)
        model = LogisticRegression(class_weight="balanced").fit(length[train], y[train])
        write_run(f"length_lr_{variant}", variant, "logistic regression on the length of the code only",
                  "reference: how far a model gets on length alone", frame, index,
                  model.predict_proba(length)[:, 1], time.time() - started, 2)

        started = time.time()
        vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", lowercase=False,
                                     ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(frame.loc[train, "code"])
        model = LogisticRegression(max_iter=4000, class_weight="balanced").fit(matrix, y[train])
        everything = vectorizer.transform(frame["code"])
        write_run(f"tfidf_lr_{variant}", variant, "TF-IDF of code tokens + class-weighted logistic regression",
                  "reference: classic text model (Lab 4 baseline)", frame, index,
                  model.predict_proba(everything)[:, 1], time.time() - started,
                  int(model.coef_.size + model.intercept_.size))
    save_log(OUT / "02_baselines_log.txt")


if __name__ == "__main__":
    main()
