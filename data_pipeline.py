import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from functools import lru_cache

import matplotlib
import numpy as np
import pandas as pd
import tree_sitter_java
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging
from tree_sitter import Language, Parser

from common import (BLUE, CONFIG, GRAY, INK, INK_2, LIGHT_GRAY, MUTED, OUT, RAW, SPLITS, SURFACE, VARIANTS,
                    WAREHOUSE, chart_style, connect, fingerprint, hub_kwargs, save_log, say, sha256_of, tidy_axes, title)

hf_logging.set_verbosity_error()

JAVA = Parser(Language(tree_sitter_java.language()))
STATUS_PARTITION = {"VULNERABLE": "P1", "FIXED": "P2", "NOT_VULNERABLE": "P3"}
PARTITIONS = [
    ("P1", "VULNERABLE", 1, "function before the fix commit"),
    ("P2", "FIXED", 0, "the same function after the fix commit"),
    ("P3", "NOT_VULNERABLE", 0, "unchanged function from a file touched by the fix"),
]
PARTITION_LABEL = {p: label for p, _, label, _ in PARTITIONS}
NVD_PLACEHOLDERS = {"NVD-CWE-noinfo": "Insufficient Information", "NVD-CWE-Other": "Other"}
MAX_LENGTH = CONFIG["model"]["max_length"]
SPLIT_RANK = {split: rank for rank, split in enumerate(SPLITS)}
LICENCE = "no licence file in the repository at the pinned commit"

SCHEMA = """
CREATE TABLE dataset_version (
    variant TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    repo_commit TEXT NOT NULL,
    archive_url TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL CHECK (length(archive_sha256) = 64),
    licence TEXT NOT NULL,
    rows_landed INTEGER NOT NULL,
    rows_clean INTEGER NOT NULL
);
CREATE TABLE label_partition (
    partition TEXT PRIMARY KEY CHECK (partition IN ('P1', 'P2', 'P3')),
    origin_status TEXT NOT NULL UNIQUE,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    meaning TEXT NOT NULL
);
CREATE TABLE cwe (
    cwe_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    mapping_usage TEXT
);
CREATE TABLE vcs_commit (
    commit_id TEXT PRIMARY KEY,
    hash_length INTEGER NOT NULL
);
CREATE TABLE java_function (
    function_id TEXT PRIMARY KEY,
    variant TEXT NOT NULL REFERENCES dataset_version(variant),
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    partition TEXT NOT NULL REFERENCES label_partition(partition),
    cwe_id TEXT NOT NULL REFERENCES cwe(cwe_id),
    commit_id TEXT NOT NULL REFERENCES vcs_commit(commit_id),
    source_idx INTEGER NOT NULL,
    function_name TEXT,
    code TEXT NOT NULL,
    code_fingerprint TEXT NOT NULL,
    n_chars INTEGER NOT NULL,
    n_tokens INTEGER NOT NULL,
    truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    review_status TEXT NOT NULL CHECK (review_status IN ('clean', 'needs_expert_review', 'duplicate')),
    source_file TEXT NOT NULL,
    source_line INTEGER NOT NULL
);
CREATE TABLE quality_issue (
    issue_id INTEGER PRIMARY KEY,
    function_id TEXT NOT NULL REFERENCES java_function(function_id),
    rule TEXT NOT NULL,
    reason TEXT NOT NULL,
    action TEXT NOT NULL
);
CREATE TABLE experiment_run (
    run_id TEXT PRIMARY KEY,
    variant TEXT NOT NULL REFERENCES dataset_version(variant),
    method TEXT NOT NULL,
    role TEXT NOT NULL,
    model_name TEXT,
    model_revision TEXT,
    class_weights TEXT,
    positive_weight REAL,
    hyperparameters TEXT NOT NULL,
    device TEXT,
    trainable_parameters INTEGER,
    train_seconds REAL,
    best_epoch INTEGER,
    threshold REAL NOT NULL,
    artifact_sha256 TEXT,
    finished_at TEXT
);
CREATE TABLE prediction (
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    function_id TEXT NOT NULL REFERENCES java_function(function_id),
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    predicted INTEGER NOT NULL CHECK (predicted IN (0, 1)),
    PRIMARY KEY (run_id, function_id)
);
CREATE TABLE metric (
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    split TEXT NOT NULL,
    threshold_rule TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, split, threshold_rule, name)
);
CREATE INDEX idx_function_variant_split ON java_function(variant, split, review_status);
CREATE VIEW model_input AS
SELECT f.function_id, f.variant, f.split, f.code, p.label, f.partition, f.cwe_id, f.n_tokens
FROM java_function f JOIN label_partition p ON p.partition = f.partition
WHERE f.review_status = 'clean';
"""
TABLES = ["dataset_version", "label_partition", "cwe", "vcs_commit", "java_function", "quality_issue",
          "experiment_run", "prediction", "metric"]


def download(url, target, expected):
    if target.exists():
        actual = sha256_of(target)
        if actual != expected:
            raise RuntimeError(f"{target.name} has SHA-256 {actual}, expected {expected}; check the file before replacing it")
        return "already on disk, checksum verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=300) as response:
        partial.write_bytes(response.read())
    actual = sha256_of(partial)
    if actual != expected:
        partial.unlink()
        raise RuntimeError(f"download from {url} has SHA-256 {actual}, expected {expected}")
    partial.rename(target)
    return "downloaded, checksum verified"


def acquire():
    title("STEP 1  ACQUISITION")
    spec = CONFIG["data"]
    sources = {}
    for variant in VARIANTS:
        archive_spec = spec["archives"][variant]
        url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{archive_spec['path']}"
        archive = RAW / f"{variant}.tar.gz"
        state = download(url, archive, archive_spec["sha256"])
        folder = RAW / variant
        if not all((folder / f"{split}.jsonl").exists() for split in SPLITS):
            folder.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(folder, filter="data")
        for path in [archive] + [folder / f"{split}.jsonl" for split in SPLITS]:
            path.chmod(0o444)
        sources[variant] = {"url": url, "sha256": archive_spec["sha256"]}
        say(f"{variant:<11} {state}; raw files set to read-only")
    catalog = spec["cwe_catalog"]
    target = RAW / f"cwec_v{catalog['version']}.xml.zip"
    state = download(catalog["url"], target, catalog["sha256"])
    target.chmod(0o444)
    sources["cwe_catalog"] = {"url": catalog["url"], "sha256": catalog["sha256"]}
    say(f"{'CWE ' + catalog['version']:<11} {state}; set to read-only")
    return sources


def extract():
    title("STEP 2  EXTRACT")
    rows = []
    for variant in VARIANTS:
        for split in SPLITS:
            with (RAW / variant / f"{split}.jsonl").open(encoding="utf-8") as fh:
                for number, line in enumerate(fh, start=1):
                    record = json.loads(line)
                    rows.append({"function_id": f"{variant}/{record['idx']}", "variant": variant, "split": split,
                                 "source_idx": record["idx"], "function_name": record["function_name"],
                                 "code": record["code"], "label": record["target"], "status": record["status"],
                                 "cwe_id": record["cwe"], "commit_id": record["commit"],
                                 "body_hash": record["body_hash"], "source_file": f"{variant}/{split}.jsonl",
                                 "source_line": number})
    raw = pd.DataFrame(rows)
    say(raw.groupby(["variant", "split"]).size().unstack().reindex(columns=SPLITS).to_string())
    say(f"function_id unique: {raw['function_id'].is_unique}")
    return raw


@lru_cache(maxsize=None)
def java_member(code):
    tree = JAVA.parse(("class Wrapper {\n" + code + "\n}").encode("utf-8"))
    if tree.root_node.has_error or not tree.root_node.children:
        return False
    body = tree.root_node.children[0].child_by_field_name("body")
    return body is not None and any(
        node.type in ("method_declaration", "constructor_declaration") for node in body.named_children)


def transform(raw, tokenizer):
    title("STEP 3  TRANSFORM AND CLEAN")
    df = raw.copy()
    audit = []

    def flag(mask, rule, reason, action):
        mask = np.asarray(mask, dtype=bool)
        for function_id in df.loc[mask, "function_id"]:
            audit.append({"function_id": function_id, "rule": rule, "reason": reason, "action": action})
        say(f"{rule}  {reason}: {int(mask.sum())} rows, {action}")

    crlf = df["code"].str.contains("\r", regex=False)
    df["code"] = df["code"].str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)
    flag(crlf, "R1", "Windows line endings", "normalised to LF, row kept")

    df["partition"] = df["status"].str.strip().str.upper().map(STATUS_PARTITION)
    expected = df["partition"].map(PARTITION_LABEL)
    invalid = df["partition"].isna() | ~df["label"].isin([0, 1]) | (expected != df["label"]) | ~df["split"].isin(SPLITS)
    flag(invalid, "R2", "label, status or split not allowed or not consistent", "quarantined")

    broken = (df["code"].str.strip() == "") | ~df["code"].map(java_member)
    flag(broken, "R3", "code is empty or does not parse as a Java method", "quarantined")

    df["code_fingerprint"] = df["code"].map(fingerprint)
    across = df.groupby("body_hash")["label"].transform("nunique") > 1
    flag(across, "R4", "same function has another label in the other dataset version", "quarantined")
    within = (df.groupby(["variant", "code_fingerprint"])["label"].transform("nunique") > 1) & ~across
    flag(within, "R4", "identical code carries both labels", "quarantined")
    df["review_status"] = np.where(invalid | broken | across | within, "needs_expert_review", "clean")

    ranked = df[df["review_status"] == "clean"].assign(rank=df["split"].map(SPLIT_RANK))
    ranked = ranked.sort_values(["variant", "code_fingerprint", "rank", "source_line"])
    repeat = ranked.duplicated(["variant", "code_fingerprint"], keep="first")
    first_split = ranked.groupby(["variant", "code_fingerprint"])["split"].transform("first")
    flag(df.index.isin(ranked.index[repeat & (ranked["split"] != first_split)]), "R5",
         "copy of a function that is already in an earlier split (leakage)", "removed from the later split")
    flag(df.index.isin(ranked.index[repeat & (ranked["split"] == first_split)]), "R5",
         "copy repeated inside the same split", "removed, first copy kept")
    df.loc[ranked.index[repeat], "review_status"] = "duplicate"

    ids = tokenizer(list(df["code"]), add_special_tokens=True, truncation=False)["input_ids"]
    df["n_tokens"] = [len(x) for x in ids]
    df["n_chars"] = df["code"].str.len()
    df["truncated"] = (df["n_tokens"] > MAX_LENGTH).astype(int)
    say()
    for variant in VARIANTS:
        part = df[df["variant"] == variant]
        counts = part["review_status"].value_counts()
        say(f"{variant:<11} {len(part)} in, {counts.get('clean', 0)} clean, "
            f"{counts.get('needs_expert_review', 0)} quarantined, {counts.get('duplicate', 0)} duplicates removed")
    return df, pd.DataFrame(audit, columns=["function_id", "rule", "reason", "action"])


def dimensions(frame, code, conflicts):
    required = frame[["code", "label", "status", "split", "cwe_id", "commit_id"]].astype(str)
    filled = required.apply(lambda column: column.str.strip() != "")
    positives = int(frame["label"].sum())
    negatives = len(frame) - positives
    valid = (frame["partition"].notna() & frame["label"].isin([0, 1]) & frame["split"].isin(SPLITS)
             & code.map(java_member))
    return {
        "rows": len(frame),
        "completeness %": round(100 * filled.to_numpy().mean(), 2),
        "uniqueness %": round(100 * (~frame.duplicated("code_fingerprint", keep=False)).mean(), 2),
        "validity %": round(100 * valid.mean(), 2),
        "label consistency %": round(100 * (~frame["function_id"].isin(conflicts)).mean(), 2),
        "functions in more than one split": int((frame.groupby("code_fingerprint")["split"].nunique() > 1).sum()),
        "rows with CRLF line endings": int(code.str.contains("\r", regex=False).sum()),
        "positives": positives,
        "negatives": negatives,
        "prevalence %": round(100 * positives / len(frame), 2),
        "negatives per positive": round(negatives / positives, 1),
    }


def assess(raw, df, audit):
    title("STEP 4  DATA QUALITY ASSESSMENT")
    conflicts = set(audit.loc[audit["rule"] == "R4", "function_id"])
    clean = df[df["review_status"] == "clean"]
    quality = {}
    for variant in VARIANTS:
        mask = (df["variant"] == variant).to_numpy()
        quality[(variant, "before")] = dimensions(df[mask], raw.loc[mask, "code"], conflicts)
        after = clean[clean["variant"] == variant]
        quality[(variant, "after")] = dimensions(after, after["code"], conflicts)
    quality = pd.DataFrame(quality)
    say(quality.to_string())

    balance = (clean.groupby(["variant", "split", "partition"]).size().unstack(fill_value=0)
               .reindex(columns=["P1", "P2", "P3"], fill_value=0))
    balance["rows"] = balance.sum(axis=1)
    balance["positives"] = balance["P1"]
    balance["prevalence %"] = (100 * balance["P1"] / balance["rows"]).round(2)
    balance = balance.reindex(pd.MultiIndex.from_product([VARIANTS, SPLITS], names=["variant", "split"]))
    say("\nClass distribution after cleaning (P1 = vulnerable, P2 = fixed, P3 = unchanged):")
    say(balance.to_string())

    lengths = clean.groupby(["variant", "partition"])["n_tokens"].agg(
        functions="size", median_tokens="median", p90_tokens=lambda s: s.quantile(0.9),
        over_512_pct=lambda s: round(100 * (s > MAX_LENGTH).mean(), 1)).round(1)
    say("\nCodeBERT token lengths:")
    say(lengths.to_string())
    issues = audit.groupby(["rule", "reason", "action"]).size().rename("rows").reset_index()

    OUT.mkdir(parents=True, exist_ok=True)
    quality.to_csv(OUT / "01_quality_before_after.csv")
    balance.to_csv(OUT / "01_class_distribution.csv")
    lengths.to_csv(OUT / "01_token_lengths.csv")
    issues.to_csv(OUT / "01_quality_issues.csv", index=False)
    audit.to_csv(OUT / "01_audit_log.csv", index=False)
    return quality, balance, lengths, issues


def build_cwe_reference():
    version = CONFIG["data"]["cwe_catalog"]["version"]
    with zipfile.ZipFile(RAW / f"cwec_v{version}.xml.zip") as archive:
        root = ET.fromstring(archive.read(f"cwec_v{version}.xml"))
    ns = {"c": root.tag[1:].split("}")[0]}
    rows = []
    for kind, path in [("weakness", "c:Weaknesses/c:Weakness"), ("category", "c:Categories/c:Category"),
                       ("view", "c:Views/c:View")]:
        for node in root.iterfind(path, ns):
            usage = node.find("c:Mapping_Notes/c:Usage", ns)
            rows.append({"cwe_id": f"CWE-{node.get('ID')}", "name": node.get("Name"), "entry_type": kind,
                         "mapping_usage": usage.text if usage is not None else None})
    for cwe_id, name in NVD_PLACEHOLDERS.items():
        rows.append({"cwe_id": cwe_id, "name": name, "entry_type": "nvd_placeholder", "mapping_usage": None})
    return pd.DataFrame(rows)


def load(df, audit, sources):
    title("STEP 5  LOAD INTO THE WAREHOUSE")
    reference = build_cwe_reference()
    missing = sorted(set(df["cwe_id"]) - set(reference["cwe_id"]))
    if missing:
        raise RuntimeError(f"CWE ids not in the MITRE catalog: {missing}")
    if WAREHOUSE.exists():
        WAREHOUSE.unlink()
    con = connect()
    con.executescript(SCHEMA)
    for variant in VARIANTS:
        part = df[df["variant"] == variant]
        con.execute("INSERT INTO dataset_version VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (variant, CONFIG["data"]["repository"], CONFIG["data"]["commit"], sources[variant]["url"],
                     sources[variant]["sha256"], LICENCE, len(part), int((part["review_status"] == "clean").sum())))
    con.executemany("INSERT INTO label_partition VALUES (?, ?, ?, ?)", PARTITIONS)
    reference.to_sql("cwe", con, if_exists="append", index=False)
    commits = pd.DataFrame({"commit_id": sorted(df["commit_id"].unique())})
    commits["hash_length"] = commits["commit_id"].str.len()
    commits.to_sql("vcs_commit", con, if_exists="append", index=False)
    columns = ["function_id", "variant", "split", "partition", "cwe_id", "commit_id", "source_idx", "function_name",
               "code", "code_fingerprint", "n_chars", "n_tokens", "truncated", "review_status", "source_file",
               "source_line"]
    df[columns].to_sql("java_function", con, if_exists="append", index=False)
    audit.to_sql("quality_issue", con, if_exists="append", index=False)
    con.commit()
    problems = con.execute("PRAGMA foreign_key_check").fetchall()
    if problems:
        raise RuntimeError(f"foreign key problems: {problems[:5]}")
    counts = {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}
    check = pd.read_sql("SELECT variant, split, COUNT(*) AS rows, SUM(label) AS positives FROM model_input "
                        "GROUP BY variant, split ORDER BY variant, split", con)
    con.close()
    WAREHOUSE.chmod(0o600)
    say(pd.Series(counts, name="rows").to_string())
    say("\nmodel_input view (what training reads):")
    say(check.to_string(index=False))
    say(f"\nwarehouse file permissions set to owner read/write only (600)")
    return counts


def draw_balance(balance):
    part = balance.loc["with_p3"]
    shares = part[["P1", "P2", "P3"]].div(part["rows"], axis=0)
    colors = {"P1": BLUE, "P2": GRAY, "P3": LIGHT_GRAY}
    names = {"P1": "P1 vulnerable (label 1)", "P2": "P2 fixed (label 0)", "P3": "P3 unchanged (label 0)"}
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 2.3), dpi=200)
        ax = fig.add_subplot()
        y = np.arange(len(SPLITS))[::-1]
        left = np.zeros(len(SPLITS))
        for key in ["P1", "P2", "P3"]:
            values = shares[key].to_numpy()
            ax.barh(y, values, left=left, height=0.55, color=colors[key], label=names[key],
                    edgecolor=SURFACE, linewidth=1.5)
            left += values
        for yi, split in zip(y, SPLITS):
            row = part.loc[split]
            ax.text(1.01, yi, f"{int(row['P1'])} of {int(row['rows']):,} vulnerable ({row['prevalence %']:.1f}%)",
                    va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, SPLITS)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("share of functions in the split")
        tidy_axes(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3)
        ax.set_title("X1 with P3 after cleaning: about one vulnerable function in 35", loc="left")
        fig.subplots_adjust(left=0.08, right=0.72, top=0.86, bottom=0.36)
        fig.savefig(OUT / "fig_class_balance.png")


def draw_lengths(df):
    clean = df[(df["review_status"] == "clean") & (df["variant"] == "with_p3")]
    styles = {"P1": (BLUE, "P1 vulnerable"), "P2": (INK_2, "P2 fixed"), "P3": (GRAY, "P3 unchanged")}
    with matplotlib.rc_context(chart_style()):
        fig = Figure(figsize=(7.2, 3.0), dpi=200)
        ax = fig.add_subplot()
        for key, (color, name) in styles.items():
            values = np.sort(clean.loc[clean["partition"] == key, "n_tokens"].to_numpy())
            share = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, share, color=color, linewidth=2, label=f"{name} (median {int(np.median(values))} tokens)")
        ax.axvline(MAX_LENGTH, color=MUTED, linewidth=0.9)
        ax.text(MAX_LENGTH * 1.06, 0.06, "CodeBERT limit, 512 tokens:\nlonger code is cut", fontsize=7.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(10, 6000)
        ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("function length in CodeBERT tokens (log scale)")
        ax.set_ylabel("share of functions this long or shorter")
        tidy_axes(ax, grid="both")
        ax.legend(loc="upper left")
        ax.set_title("Unchanged functions (P3) are much shorter than vulnerable and fixed ones", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "fig_token_lengths.png")


def summarise(raw, df, audit, quality, balance, counts, sources):
    clean = df[df["review_status"] == "clean"]
    issues = audit.groupby(["rule", "reason"]).size()
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "rows_landed": raw.groupby("variant").size().to_dict(),
        "rows_clean": clean.groupby("variant").size().to_dict(),
        "rows_by_status": df.groupby(["variant", "review_status"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "issues": {f"{rule} {reason}": int(n) for (rule, reason), n in issues.items()},
        "quality": {f"{v} {stage}": quality[(v, stage)].to_dict() for v, stage in quality.columns},
        "class_distribution": {f"{v} {s}": balance.loc[(v, s)].to_dict() for v, s in balance.index},
        "truncated_pct_with_p3": round(100 * clean.loc[clean["variant"] == "with_p3", "truncated"].mean(), 1),
        "warehouse_rows": counts,
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    return summary


def main():
    sources = acquire()
    raw = extract()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"], **hub_kwargs())
    df, audit = transform(raw, tokenizer)
    quality, balance, lengths, issues = assess(raw, df, audit)
    counts = load(df, audit, sources)
    draw_balance(balance)
    draw_lengths(df)
    summarise(raw, df, audit, quality, balance, counts, sources)
    save_log(OUT / "01_data_pipeline_log.txt")
    return df, audit


if __name__ == "__main__":
    main()
