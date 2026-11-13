"""Training, evaluation and prediction for the phishing email classifier."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")


def clean_text(text: str) -> str:
    """Lowercase and replace URLs, emails and numbers with placeholder tokens.

    Placeholders keep the *signal* (an email contains a link) without the model
    memorising specific domains that won't appear in future phishing emails.
    """
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str, text_col: str, label_col: str, positive_label: str) -> tuple[pd.Series, np.ndarray]:
    """Load a CSV and return (cleaned texts, labels as 1 = phishing, 0 = legitimate)."""
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Columns {missing} not in CSV. Available: {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    X = df[text_col].map(clean_text)
    y = (df[label_col].astype(str).str.strip() == str(positive_label)).astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Only one class found. Is --positive-label '{positive_label}' right? "
                         f"Labels seen: {sorted(df[label_col].astype(str).unique())[:10]}")
    return X, y


def build_models() -> dict[str, Pipeline]:
    tfidf = dict(ngram_range=(1, 2), stop_words="english", min_df=1, max_df=0.95, sublinear_tf=True, max_features=50000)
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", MultinomialNB()),
        ]),
    }


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]  # [[TN, FP], [FN, TP]]

    def as_row(self) -> dict:
        (tn, fp), (fn, tp) = self.confusion
        return {"accuracy": round(self.accuracy, 3), "precision": round(self.precision, 3),
                "recall": round(self.recall, 3), "f1": round(self.f1, 3),
                "false_negatives": fn, "false_positives": fp}


def evaluate(model: Pipeline, X_test, y_test) -> Metrics:
    pred = model.predict(X_test)
    return Metrics(
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    )


def train_and_compare(X, y, test_size: float = 0.2, seed: int = 42):
    """Train every model on the same split. Returns (fitted models, metrics, best name)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    models, results = build_models(), {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
    # Pick on recall first (missing phishing is the costly error), then F1 as tie-break.
    best = max(results, key=lambda n: (round(results[n].recall, 3), results[n].f1))
    return models, results, best


def top_features(model: Pipeline, n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Words that push hardest towards phishing / legitimate (logistic regression only)."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    names = model.named_steps["tfidf"].get_feature_names_out()
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    return {
        "phishing": [(names[i], round(float(coefs[i]), 3)) for i in order[::-1][:n]],
        "legitimate": [(names[i], round(float(coefs[i]), 3)) for i in order[:n]],
    }


def explain(model: Pipeline, text: str, n: int = 5) -> list[tuple[str, float]]:
    """Words in *this* email that contributed most towards a phishing verdict."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    vec = model.named_steps["tfidf"].transform([clean_text(text)])
    names = model.named_steps["tfidf"].get_feature_names_out()
    contrib = vec.multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: p[1], reverse=True)
    return [(names[i], round(float(v), 3)) for i, v in pairs[:n] if v > 0]


def predict(model: Pipeline, text: str) -> tuple[str, float]:
    """Return (verdict, probability of phishing)."""
    prob = float(model.predict_proba([clean_text(text)])[0][1])
    return ("PHISHING" if prob >= 0.5 else "LEGITIMATE"), prob
