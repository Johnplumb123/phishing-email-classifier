"""Train and compare models, save the best one, and write a results report.

Example (with the Kaggle dataset, see README):
    python train.py --data data/Phishing_Email.csv --text-col "Email Text" \
        --label-col "Email Type" --positive-label "Phishing Email"

Quick smoke test with the bundled toy sample:
    python train.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phishclf.model import load_dataset, top_features, train_and_compare  # noqa: E402


def plot_confusion(confusion, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(confusion, cmap="Blues")
    labels = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    peak = max(max(row) for row in confusion) or 1
    for i in range(2):
        for j in range(2):
            v = confusion[i][j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > peak / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sample_emails.csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--positive-label", default="phishing", help="Label value meaning phishing")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    X, y = load_dataset(args.data, args.text_col, args.label_col, args.positive_label)
    print(f"Loaded {len(y)} emails: {int(y.sum())} phishing, {int(len(y) - y.sum())} legitimate\n")

    models, results, best = train_and_compare(X, y)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print(f"{'model':<22}{'acc':>7}{'prec':>7}{'recall':>8}{'f1':>7}{'FN':>6}{'FP':>6}")
    for name, m in results.items():
        r = m.as_row()
        print(f"{name:<22}{r['accuracy']:>7}{r['precision']:>7}{r['recall']:>8}{r['f1']:>7}"
              f"{r['false_negatives']:>6}{r['false_positives']:>6}")
        plot_confusion(m.confusion, name.replace("_", " ").title(), out / f"confusion_{name}.png")

    print(f"\nBest model (by recall, then F1): {best}")
    joblib.dump(models[best], out / "model.joblib")
    # Always keep the logistic regression too, since it can explain its predictions
    joblib.dump(models["logistic_regression"], out / "model_lr.joblib")

    feats = top_features(models["logistic_regression"])
    if feats:
        print("\nWords most associated with phishing:")
        print("  " + ", ".join(w for w, _ in feats["phishing"]))

    report = {"dataset": args.data, "n_emails": int(len(y)), "best_model": best,
              "metrics": {n: m.as_row() for n, m in results.items()}, "top_features": feats}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model, confusion matrices and metrics.json to {out}/")


if __name__ == "__main__":
    main()
