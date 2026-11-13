"""Classify a single email from the command line.

    python predict.py "Your account has been suspended, verify now at http://..."
    python predict.py --file suspicious_email.txt
"""
import argparse
import sys

import joblib

from phishclf.model import explain, predict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", help="Email text (or use --file / stdin)")
    p.add_argument("--file")
    p.add_argument("--model", default="results/model_lr.joblib")
    args = p.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    model = joblib.load(args.model)
    verdict, prob = predict(model, text)
    print(f"{verdict}  (phishing probability: {prob:.1%})")
    reasons = explain(model, text)
    if reasons:
        print("Top signals: " + ", ".join(w for w, _ in reasons))


if __name__ == "__main__":
    main()
