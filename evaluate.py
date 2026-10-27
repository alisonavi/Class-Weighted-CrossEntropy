import json
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from common import (AQUA, BLUE, CONFIG, GRAY, INK, INK_2, LIGHT_BLUE, LIGHT_GRAY, MUTED, ORANGE, OUT, RUNS, SEED, SURFACE,
                    WAREHOUSE, chart_style, connect, save_log, say, tidy_axes, title)

ORDER = ["codebert_weighted", "codebert_plain", "codebert_sqrt", "frozen_codebert_lr", "tfidf_lr_with_p3",
         "length_lr_with_p3", "majority_with_p3", "codebert_weighted_balanced_set", "tfidf_lr_without_p3",
         "length_lr_without_p3", "majority_without_p3"]
NAMES = {
    "codebert_weighted": "CodeBERT + class-weighted CE (primary)",
    "codebert_plain": "CodeBERT + plain CE",
    "codebert_sqrt": "CodeBERT + square-root class weights",
    "frozen_codebert_lr": "Frozen CodeBERT + weighted LR (alternative)",
    "tfidf_lr_with_p3": "TF-IDF + weighted LR",
    "length_lr_with_p3": "Code length only",
    "majority_with_p3": "Always \"not vulnerable\"",
    "codebert_weighted_balanced_set": "CodeBERT + class-weighted CE (primary)",
    "tfidf_lr_without_p3": "TF-IDF + weighted LR",
    "length_lr_without_p3": "Code length only",
    "majority_without_p3": "Always \"not vulnerable\"",
}
COLORS = {"codebert_weighted": BLUE, "codebert_plain": ORANGE, "codebert_sqrt": LIGHT_BLUE, "frozen_codebert_lr": AQUA,
          "tfidf_lr_with_p3": GRAY, "length_lr_with_p3": LIGHT_GRAY}
PAPER = {
    "with_p3": [("ContraBERT, reported in the paper", 0.85), ("WizardCoder 13B, reported", 0.86),
                ("WizardCoder + focal loss + sample weights, reported", 0.878)],
    "without_p3": [("ContraBERT, reported in the paper", 0.66), ("WizardCoder 13B, reported", 0.69)],
}
PAIRS = [("codebert_weighted", "codebert_plain"), ("codebert_weighted", "codebert_sqrt"),
         ("codebert_weighted", "frozen_codebert_lr"), ("codebert_weighted", "tfidf_lr_with_p3"),
         ("codebert_weighted", "length_lr_with_p3")]
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
TARGET = CONFIG["evaluation"]["accuracy_target"]


def runs_variant(name):
    return "without_p3" if name.endswith("without_p3") or name.endswith("balanced_set") else "with_p3"


def load_runs():
    runs = {}
    for name in ORDER:
        folder = RUNS / name
        if (folder / "predictions.csv").exists() and (folder / "run_summary.json").exists():
            runs[name] = {"summary": json.loads((folder / "run_summary.json").read_text()),
                          "predictions": pd.read_csv(folder / "predictions.csv")}
    return runs


def tuned_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def target_threshold(y, p, target):
    candidates = np.unique(p)
    for threshold in candidates:
        if ((p >= threshold).astype(int) == y).mean() >= target:
            return float(threshold)
    return float(candidates[-1]) + 1e-9


def scores(y, p, threshold):
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "flagged": int(tp + fp),
    }


def resample_stats(y, p, threshold, draws):
    out = np.zeros((len(draws), 3))
    for k, index in enumerate(draws):
        yb, pb = y[index], p[index]
        predicted = pb >= threshold
        tp = int((predicted & (yb == 1)).sum())
        fp = int((predicted & (yb == 0)).sum())
        fn = int((~predicted & (yb == 1)).sum())
        out[k] = [roc_auc_score(yb, pb), average_precision_score(yb, pb), 2 * tp / (2 * tp + fp + fn) if tp else 0.0]
    return out


def evaluate_runs(runs):
    title("STEP 1  TEST METRICS (threshold tuned on validation for the best F1)")
    rng = np.random.default_rng(SEED)
    n_draws = CONFIG["evaluation"]["bootstrap_samples"]
    draws, reference_ids, rows, default_rows, resampled, test_frames = {}, {}, [], [], {}, {}
    for name, run in runs.items():
        frame = run["predictions"]
        variant = run["summary"]["variant"]
        valid = frame[frame["split"] == "valid"]
        test = frame[frame["split"] == "test"].sort_values("function_id").reset_index(drop=True)
        test_frames[name] = test
        fixed = run["summary"].get("fixed_threshold")
        threshold = fixed if fixed is not None else tuned_threshold(valid["label"].to_numpy(), valid["probability"].to_numpy())
        y, p = test["label"].to_numpy(), test["probability"].to_numpy()
        if variant not in draws:
            draws[variant] = [rng.integers(0, len(y), len(y)) for _ in range(n_draws)]
            draws[variant] = [d for d in draws[variant] if 0 < y[d].sum() < len(d)]
            reference_ids[variant] = test["function_id"].to_numpy()
        if not np.array_equal(reference_ids[variant], test["function_id"].to_numpy()):
            raise RuntimeError(f"{name} was scored on a different test set")
        row = {"run": name, "variant": variant, "method": NAMES[name], "role": run["summary"]["role"], **scores(y, p, threshold)}
        stats = resample_stats(y, p, threshold, draws[variant])
        resampled[name] = stats
        for k, metric in enumerate(["roc_auc", "pr_auc", "f1"]):
            row[f"{metric}_low"], row[f"{metric}_high"] = np.percentile(stats[:, k], [2.5, 97.5])
        hard = test["partition"].isin(["P1", "P2"]).to_numpy()
        row["roc_auc_p1_vs_p2"] = roc_auc_score(y[hard], p[hard]) if variant == "with_p3" else row["roc_auc"]
        easy = test["partition"].isin(["P1", "P3"]).to_numpy()
        row["roc_auc_p1_vs_p3"] = roc_auc_score(y[easy], p[easy]) if variant == "with_p3" else np.nan
        row["meets_accuracy_target"] = row["accuracy"] >= TARGET
        row["train_seconds"] = run["summary"].get("train_seconds")
        row["trainable_parameters"] = run["summary"].get("trainable_parameters")
        rows.append(row)
        target = fixed if fixed is not None else target_threshold(valid["label"].to_numpy(), valid["probability"].to_numpy(), TARGET)
        for rule, cut in [("best F1 on validation", threshold), (f"accuracy {TARGET:.0%} on validation", target), ("default 0.5", 0.5)]:
            default_rows.append({"run": name, "variant": variant, "method": NAMES[name], "rule": rule, **scores(y, p, cut)})
    table = pd.DataFrame(rows)
    points = pd.DataFrame(default_rows)
    points["meets_accuracy_target"] = points["accuracy"] >= TARGET
    shown = ["run", "accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_low", "roc_auc_high", "pr_auc",
             "roc_auc_p1_vs_p2", "threshold", "tp", "fp", "fn", "tn"]
    say(table[shown].round(4).to_string(index=False))
    say(f"\nThree operating points for each model (test split; every threshold comes from validation or is fixed):")
    say(points[["run", "rule", "threshold", "accuracy", "precision", "recall", "f1", "flagged", "meets_accuracy_target"]]
        .round(4).to_string(index=False))
    table.to_csv(OUT / "02_metrics_test.csv", index=False)
    points.to_csv(OUT / "02_operating_points.csv", index=False)
    return table, points, resampled, test_frames


def paired(table, resampled):
    title("STEP 2  PAIRED BOOTSTRAP: is the difference bigger than the noise?")
    rows = []
    for a, b in PAIRS:
        if a not in resampled or b not in resampled:
            continue
        for k, metric in enumerate(["roc_auc", "pr_auc", "f1"]):
            diff = resampled[a][:, k] - resampled[b][:, k]
            low, high = np.percentile(diff, [2.5, 97.5])
            observed = float(table.set_index("run").loc[a, metric] - table.set_index("run").loc[b, metric])
            rows.append({"run_a": a, "run_b": b, "metric": metric, "difference": observed, "low": low, "high": high,
                         "share_a_better": float((diff > 0).mean()), "clear": bool(low > 0 or high < 0)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        say(frame.round(4).to_string(index=False))
        frame.to_csv(OUT / "02_paired_bootstrap.csv", index=False)
    return frame


def review_budget(test_frames):
    title("STEP 3  REVIEW BUDGET: share of vulnerable functions found when reviewing the top-scored code")
    rows = []
    for name, test in test_frames.items():
        if name.startswith("majority") or runs_variant(name) != "with_p3":
            continue
        ranked = test.sort_values("probability", ascending=False, kind="stable")
        positives = ranked["label"].sum()
        for budget in BUDGETS:
            top = ranked.head(int(round(budget * len(ranked))))
            rows.append({"run": name, "budget": budget, "reviewed": len(top), "found": int(top["label"].sum()),
                         "recall": top["label"].sum() / positives, "precision": top["label"].mean()})
    frame = pd.DataFrame(rows)
    say(frame.pivot(index="run", columns="budget", values="recall").round(3).to_string())
    frame.to_csv(OUT / "02_review_budget.csv", index=False)
    return frame


def write_warehouse(runs, table):
    title("STEP 4  RESULTS INTO THE WAREHOUSE")
    con = connect()
    for name in ["metric", "prediction", "experiment_run"]:
        con.execute(f"DELETE FROM {name}")
    for name, run in runs.items():
        s = run["summary"]
        row = table.set_index("run").loc[name]
        con.execute("INSERT INTO experiment_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            name, s["variant"], s["method"], s["role"], s.get("model", {}).get("name"),
            s.get("model", {}).get("revision"), s.get("class_weights_rule"),
            (s.get("class_weights") or {}).get("vulnerable"), json.dumps(s.get("hyperparameters", {})), s.get("device"),
            s.get("trainable_parameters"), s.get("train_seconds"), s.get("best_epoch"), float(row["threshold"]),
            s.get("checkpoint_sha256"), s.get("finished_at")))
        frame = run["predictions"]
        con.executemany("INSERT INTO prediction VALUES (?, ?, ?, ?)", [
            (name, fid, float(p), int(p >= row["threshold"])) for fid, p in zip(frame["function_id"], frame["probability"])])
        metrics = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
        con.executemany("INSERT INTO metric VALUES (?, ?, ?, ?, ?)", [
            (name, "test", "max F1 on validation", m, float(row[m])) for m in metrics])
    con.commit()
    query = """
        SELECT f.cwe_id, c.name AS cwe_name, COUNT(*) AS vulnerable, SUM(p.predicted) AS caught,
               ROUND(1.0 * SUM(p.predicted) / COUNT(*), 3) AS recall
        FROM prediction p
        JOIN java_function f ON f.function_id = p.function_id
        JOIN cwe c ON c.cwe_id = f.cwe_id
        WHERE p.run_id = ? AND f.split = 'test' AND f.partition = 'P1'
        GROUP BY f.cwe_id, c.name
        HAVING COUNT(*) >= 4
        ORDER BY vulnerable DESC, f.cwe_id
    """
    cwe = pd.read_sql(query, con, params=["codebert_weighted"]) if "codebert_weighted" in runs else pd.DataFrame()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ["experiment_run", "prediction", "metric"]}
    con.close()
    say(pd.Series(counts, name="rows").to_string())
    if not cwe.empty:
        say("\nRecall by CWE for the primary model on the test split (SQL over the warehouse, CWEs with 4+ cases):")
        say(cwe.to_string(index=False))
        cwe.to_csv(OUT / "02_cwe_recall.csv", index=False)
    return cwe


def draw_training_curves(runs):
    names = [n for n in ["codebert_weighted", "codebert_plain", "codebert_sqrt"] if n in runs]
    if not names:
        return
    colors = {"codebert_weighted": BLUE, "codebert_plain": ORANGE, "codebert_sqrt": LIGHT_BLUE}
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 2.9), dpi=200)
        left, right = fig.subplots(1, 2)
        for name in names:
            log = pd.read_csv(RUNS / name / "train_log.csv")
            best = log.loc[log["valid_roc_auc"].idxmax()]
            left.plot(log["epoch"], log["valid_roc_auc"], color=colors[name], linewidth=2, marker="o", markersize=4,
                      label=NAMES[name])
            left.plot([best["epoch"]], [best["valid_roc_auc"]], "o", markersize=9, markerfacecolor="none",
                      markeredgecolor=colors[name], markeredgewidth=1.4)
            right.plot(log["epoch"], log["valid_log_loss"], color=colors[name], linewidth=2, marker="o", markersize=4)
        left.set_title("Validation ROC AUC (circle = kept checkpoint)", loc="left", fontsize=9)
        right.set_title("Validation log loss (lower is better)", loc="left", fontsize=9)
        for ax in (left, right):
            ax.set_xlabel("epoch")
            ax.set_xticks(range(1, int(max(pd.read_csv(RUNS / n / "train_log.csv")["epoch"].max() for n in names)) + 1))
            tidy_axes(ax)
        fig.legend(*left.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(OUT / "fig_training_curves.png")


def draw_roc_pr(test_frames, table):
    names = [n for n in ["codebert_weighted", "codebert_plain", "frozen_codebert_lr", "tfidf_lr_with_p3",
                         "length_lr_with_p3"] if n in test_frames]
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 3.3), dpi=200)
        roc_ax, pr_ax = fig.subplots(1, 2)
        prevalence = test_frames[names[0]]["label"].mean()
        roc_ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8)
        pr_ax.axhline(prevalence, color=MUTED, linewidth=0.8)
        pr_ax.text(0.5, prevalence + 0.012, f"random guess = {prevalence:.3f}", ha="center", fontsize=7, color=INK_2)
        for name in reversed(names):
            test = test_frames[name]
            row = table.set_index("run").loc[name]
            fpr, tpr, _ = roc_curve(test["label"], test["probability"])
            precision, recall, _ = precision_recall_curve(test["label"], test["probability"])
            width = 2.2 if name == "codebert_weighted" else 1.5
            roc_ax.plot(fpr, tpr, color=COLORS[name], linewidth=width, label=f"{NAMES[name]}  {row['roc_auc']:.3f}")
            pr_ax.plot(recall, precision, color=COLORS[name], linewidth=width, label=f"{NAMES[name]}  {row['pr_auc']:.3f}")
        roc_ax.set_xlabel("false positive rate")
        roc_ax.set_ylabel("true positive rate (recall)")
        roc_ax.set_title("ROC curve, test split", loc="left", fontsize=9)
        pr_ax.set_xlabel("recall")
        pr_ax.set_ylabel("precision")
        pr_ax.set_title("Precision-recall curve, test split", loc="left", fontsize=9)
        pr_ax.set_ylim(0, 1)
        for ax in (roc_ax, pr_ax):
            ax.set_xlim(0, 1)
            tidy_axes(ax, grid="both")
        handles, labels = roc_ax.get_legend_handles_labels()
        fig.legend(handles[::-1], [l.split("  ")[0] for l in labels[::-1]], loc="lower center", ncol=3, fontsize=7.2)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(OUT / "fig_roc_pr.png")


def draw_confusion(table):
    names = [n for n in ["codebert_weighted", "codebert_plain", "frozen_codebert_lr"] if n in set(table["run"])]
    if not names:
        return
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 2.5), dpi=200)
        axes = np.atleast_1d(fig.subplots(1, len(names)))
        for ax, name in zip(axes, names):
            row = table.set_index("run").loc[name]
            cells = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
            share = cells / cells.sum(axis=1, keepdims=True)
            shades = np.array([["#f0efec", "#cde2fb"], ["#cde2fb", "#86b6ef"]])
            for i in range(2):
                for j in range(2):
                    ax.add_patch(matplotlib.patches.Rectangle((j, 1 - i), 0.97, 0.97, color=shades[i][j]))
                    ax.text(j + 0.485, 1 - i + 0.56, f"{int(cells[i, j]):,}", ha="center", va="center", fontsize=11, color=INK)
                    ax.text(j + 0.485, 1 - i + 0.3, f"{share[i, j]:.1%} of row", ha="center", va="center", fontsize=7, color=INK_2)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks([0.485, 1.485], ["predicted\nnot vulnerable", "predicted\nvulnerable"], fontsize=7)
            ax.set_yticks([1.485, 0.485], ["not\nvulnerable", "vulnerable"], fontsize=7)
            ax.set_title(f"{NAMES[name]}\nthreshold {row['threshold']:.3f}", loc="left", fontsize=7.5)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(OUT / "fig_confusion.png")


def draw_review_budget(test_frames):
    names = [n for n in ["codebert_weighted", "codebert_plain", "frozen_codebert_lr", "tfidf_lr_with_p3", "length_lr_with_p3"]
             if n in test_frames]
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 3.6), dpi=200)
        ax = fig.add_subplot()
        ax.plot([0, 0.3], [0, 0.3], color=MUTED, linewidth=0.8)
        ax.text(0.262, 0.205, "random order", fontsize=7.5, color=INK_2, rotation=13)
        for name in reversed(names):
            ranked = test_frames[name].sort_values("probability", ascending=False, kind="stable")
            found = np.concatenate([[0], np.cumsum(ranked["label"].to_numpy())]) / ranked["label"].sum()
            reviewed = np.arange(len(found)) / len(ranked)
            ax.plot(reviewed, found, color=COLORS[name], linewidth=2.2 if name == "codebert_weighted" else 1.5,
                    label=NAMES[name])
        primary = test_frames.get("codebert_weighted")
        if primary is not None:
            ranked = primary.sort_values("probability", ascending=False, kind="stable")
            spots = {0.05: (0.004, 0.66), 0.10: (0.122, 0.21)}
            for budget, place in spots.items():
                share = ranked.head(int(round(budget * len(ranked))))["label"].sum() / ranked["label"].sum()
                ax.plot([budget], [share], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
                ax.annotate(f"primary model, top {budget:.0%} read:\n{share:.0%} of vulnerable ones found", (budget, share),
                            xytext=place, fontsize=7.5, color=INK,
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("share of test functions a reviewer reads, highest score first")
        ax.set_ylabel("share of vulnerable functions found")
        tidy_axes(ax, grid="both")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=3, fontsize=7.2)
        ax.set_title("What a review team gets for its effort (test split, X1 with P3)", loc="left")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(OUT / "fig_review_budget.png")


def draw_partition_scores(test_frames, table):
    if "codebert_weighted" not in test_frames:
        return
    test = test_frames["codebert_weighted"]
    threshold = table.set_index("run").loc["codebert_weighted", "threshold"]
    styles = {"P1": (BLUE, "P1 vulnerable"), "P2": (INK_2, "P2 fixed version of the same function"),
              "P3": (GRAY, "P3 unchanged function")}
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 3.0), dpi=200)
        ax = fig.add_subplot()
        for key, (color, name) in styles.items():
            values = np.sort(test.loc[test["partition"] == key, "probability"].to_numpy())
            share = np.arange(1, len(values) + 1) / len(values)
            above = (values >= threshold).mean()
            ax.plot(values, share, color=color, linewidth=2, label=f"{name}: {above:.0%} flagged (n = {len(values):,})")
        ax.axvline(threshold, color=MUTED, linewidth=0.9)
        ax.text(threshold * 1.08, 0.04, f"threshold {threshold:.3f}:\nflagged to the right", fontsize=7.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(5e-4, 1)
        ax.set_xticks([0.001, 0.01, 0.1, 0.5, 1], ["0.001", "0.01", "0.1", "0.5", "1"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("score from the primary model (log scale)")
        ax.set_ylabel("share of functions with this score or lower")
        tidy_axes(ax, grid="both")
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_title("The model separates untouched code from fix commits, not the bug from its fix", loc="left",
                     fontsize=9.5)
        fig.tight_layout()
        fig.savefig(OUT / "fig_partition_scores.png")


def draw_comparison(table):
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 4.6), dpi=200)
        axes = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [7.5, 5]})
        for ax, variant, heading in zip(axes, ["with_p3", "without_p3"],
                                        ["Imbalanced: X1 with P3 (1 vulnerable in 35)", "Balanced: X1 without P3 (P1 vs P2 only)"]):
            part = table[table["variant"] == variant].set_index("run")
            labels, values, lows, highs, colors = [], [], [], [], []
            for name in [n for n in ORDER if n in part.index and not n.startswith("majority")]:
                row = part.loc[name]
                labels.append(NAMES[name])
                values.append(row["roc_auc"])
                lows.append(row["roc_auc"] - row["roc_auc_low"])
                highs.append(row["roc_auc_high"] - row["roc_auc"])
                colors.append(COLORS.get(name, BLUE if name.startswith("codebert") else GRAY))
            if variant == "with_p3" and "codebert_weighted" in part.index:
                labels.append("Primary model, scored on P1 vs P2 only")
                values.append(part.loc["codebert_weighted", "roc_auc_p1_vs_p2"])
                lows.append(0)
                highs.append(0)
                colors.append(BLUE)
            paper = PAPER[variant]
            y = np.arange(len(labels) + len(paper) + 1)[::-1]
            for i, (value, low, high, color) in enumerate(zip(values, lows, highs, colors)):
                hollow = labels[i].startswith("Primary model, scored")
                ax.errorbar([value], [y[i]], xerr=[[low], [high]], fmt="o", color=color, ecolor=color, elinewidth=1.2,
                            capsize=2.5, markersize=6, markerfacecolor=SURFACE if hollow else color, markeredgewidth=1.4)
                ax.text(value + high + 0.012, y[i], f"{value:.3f}", va="center", fontsize=7.5, color=INK)
            for j, (label, value) in enumerate(paper):
                yi = y[len(labels) + 1 + j]
                ax.plot([value], [yi], "D", markerfacecolor=SURFACE, markeredgecolor=INK_2, markersize=5.5, markeredgewidth=1.3)
                ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=7.5, color=INK_2)
            ticks = labels + [""] + [p[0] for p in paper]
            ax.set_yticks(y, ticks, fontsize=7.5)
            ax.axvline(0.5, color=MUTED, linewidth=0.8)
            ax.set_title(heading, loc="left", fontsize=9)
            tidy_axes(ax, grid="x")
            ax.spines["left"].set_visible(False)
        axes[1].set_xlim(0.4, 1.0)
        axes[1].set_xlabel("test ROC AUC   ·   line: 95% bootstrap interval   ·   diamond: reported by Shestov et al.")
        axes[1].text(0.503, axes[1].get_ylim()[0] + 0.2, "chance", fontsize=7, color=INK_2)
        fig.tight_layout()
        fig.savefig(OUT / "fig_method_comparison.png")


def summarise(table, default, pairs, budget, cwe):
    def pick(name):
        if name not in set(table["run"]):
            return None
        row = table.set_index("run").loc[name]
        keys = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "roc_auc_low", "roc_auc_high",
                "pr_auc", "pr_auc_low", "pr_auc_high", "f1_low", "f1_high", "roc_auc_p1_vs_p2", "roc_auc_p1_vs_p3",
                "threshold", "tp", "fp", "fn", "tn", "flagged", "train_seconds", "trainable_parameters"]
        return {k: (None if pd.isna(row[k]) else (float(row[k]) if not isinstance(row[k], (int, np.integer)) else int(row[k])))
                for k in keys}

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accuracy_target": TARGET,
        "runs": {name: pick(name) for name in ORDER if pick(name) is not None},
        "operating_points": {run: {r["rule"]: {k: float(r[k]) for k in ["threshold", "accuracy", "precision", "recall", "f1"]}
                                    | {"flagged": int(r["flagged"]), "tp": int(r["tp"])} for _, r in part.iterrows()}
                              for run, part in default.groupby("run")},
        "paired": pairs.to_dict(orient="records") if not pairs.empty else [],
        "review_budget": budget.to_dict(orient="records"),
        "cwe_recall": cwe.to_dict(orient="records") if not cwe.empty else [],
        "paper": PAPER,
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main():
    runs = load_runs()
    say(f"runs found: {', '.join(runs)}")
    table, default, resampled, test_frames = evaluate_runs(runs)
    pairs = paired(table, resampled)
    budget = review_budget(test_frames)
    cwe = write_warehouse(runs, table)
    draw_training_curves(runs)
    draw_roc_pr(test_frames, table)
    draw_confusion(table)
    draw_review_budget(test_frames)
    draw_partition_scores(test_frames, table)
    draw_comparison(table)
    summary = summarise(table, default, pairs, budget, cwe)
    save_log(OUT / "02_evaluation_log.txt")
    return summary


if __name__ == "__main__":
    main()
