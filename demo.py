import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common import BASE, CONFIG, OUT, RUNS, SEED, connect, device, sha256_of

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

EXAMPLES = {
    "login, SQL built by hand": """public User findUser(Connection db, String name) throws SQLException {
    Statement st = db.createStatement();
    ResultSet rs = st.executeQuery("SELECT * FROM users WHERE name = '" + name + "'");
    return rs.next() ? new User(rs) : null;
}""",
    "login, prepared statement": """public User findUser(Connection db, String name) throws SQLException {
    PreparedStatement st = db.prepareStatement("SELECT * FROM users WHERE name = ?");
    st.setString(1, name);
    ResultSet rs = st.executeQuery();
    return rs.next() ? new User(rs) : null;
}""",
    "simple getter": """public String getName() {
    return name;
}""",
}


def load(run):
    summary = json.loads((RUNS / run / "run_summary.json").read_text())
    folder = BASE / summary["checkpoint"]
    actual = sha256_of(folder / "model.safetensors")
    if actual != summary["checkpoint_sha256"]:
        sys.exit(f"refusing to load {folder}: SHA-256 {actual[:16]}... does not match run_summary.json")
    threshold = json.loads((OUT / "metrics.json").read_text())["runs"][run]["threshold"]
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).to(device()).eval()
    return tokenizer, model, threshold, actual


@torch.no_grad()
def score(tokenizer, model, code):
    batch = tokenizer(code, truncation=True, max_length=CONFIG["model"]["max_length"], return_tensors="pt")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    return torch.softmax(model(**batch).logits.float(), dim=-1)[0, 1].item()


def test_sample(n):
    con = connect(read_only=True)
    frame = pd.read_sql("SELECT m.function_id, f.function_name, m.cwe_id, m.partition, m.label, m.code "
                        "FROM model_input m JOIN java_function f ON f.function_id = m.function_id "
                        "WHERE m.variant = 'with_p3' AND m.split = 'test'", con)
    con.close()
    picked = frame.groupby("partition", group_keys=False).sample(n=n, random_state=SEED)
    return picked.sort_values(["partition", "function_id"])


def main():
    parser = argparse.ArgumentParser(description="Score Java functions with the trained model")
    parser.add_argument("files", nargs="*", help="Java files, each holding one method")
    parser.add_argument("--run", default="codebert_weighted")
    parser.add_argument("--sample", type=int, default=0, help="also score N test functions from each partition")
    args = parser.parse_args()
    tokenizer, model, threshold, digest = load(args.run)
    print(f"model {args.run}: checkpoint SHA-256 {digest[:16]}... verified, threshold {threshold:.3f}\n")
    items = [(Path(p).name, Path(p).read_text()) for p in args.files] or list(EXAMPLES.items())
    for name, code in items:
        p = score(tokenizer, model, code)
        print(f"{name:<28} score {p:.3f}  {'FLAG for review' if p >= threshold else 'no flag'}")
    if args.sample:
        print(f"\nReal test functions (never seen in training):")
        for row in test_sample(args.sample).itertuples():
            p = score(tokenizer, model, row.code)
            truth = "vulnerable" if row.label == 1 else "not vulnerable"
            print(f"{row.partition}  {row.function_name[:26]:<26} {row.cwe_id:<15} truth {truth:<15} "
                  f"score {p:.3f}  {'FLAG' if p >= threshold else 'no flag'}")


if __name__ == "__main__":
    main()
