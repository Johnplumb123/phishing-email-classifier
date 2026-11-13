from pathlib import Path

import pandas as pd
import pytest

from phishclf.model import (
    clean_text, explain, load_dataset, predict, top_features, train_and_compare,
)

SAMPLE = Path(__file__).parent.parent / "data" / "sample_emails.csv"


@pytest.fixture(scope="module")
def trained():
    X, y = load_dataset(str(SAMPLE), "text", "label", "phishing")
    return train_and_compare(X, y, test_size=0.3)


def test_clean_text_replaces_urls_emails_and_numbers():
    out = clean_text("Pay £50 at http://evil.com/x or email scam@bad.com")
    assert "urltoken" in out and "emailtoken" in out and "numtoken" in out
    assert "evil.com" not in out and "50" not in out


def test_load_dataset_maps_labels():
    X, y = load_dataset(str(SAMPLE), "text", "label", "phishing")
    assert len(X) == len(y)
    assert set(y) == {0, 1}


def test_load_dataset_rejects_wrong_columns():
    with pytest.raises(ValueError, match="not in CSV"):
        load_dataset(str(SAMPLE), "body", "label", "phishing")


def test_load_dataset_rejects_wrong_positive_label(tmp_path):
    with pytest.raises(ValueError, match="Only one class"):
        load_dataset(str(SAMPLE), "text", "label", "spam")


def test_load_dataset_drops_empty_rows(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame({"text": ["click http://x.com now", None, "  ", "see you at lunch"],
                  "label": ["phishing", "phishing", "legitimate", "legitimate"]}).to_csv(csv, index=False)
    X, y = load_dataset(str(csv), "text", "label", "phishing")
    assert len(X) == 2


def test_both_models_are_trained_and_scored(trained):
    models, results, best = trained
    assert set(results) == {"logistic_regression", "naive_bayes"}
    assert best in results
    for m in results.values():
        assert 0 <= m.recall <= 1
        assert sum(sum(r) for r in m.confusion) > 0


def test_obvious_examples_are_classified_correctly(trained):
    model = trained[0]["logistic_regression"]
    verdict, prob = predict(model, "URGENT your account is suspended, verify your password at http://x.co now")
    assert verdict == "PHISHING" and prob > 0.5
    verdict, _ = predict(model, "Hi team, notes from the sprint review are attached, see you Friday")
    assert verdict == "LEGITIMATE"


def test_explanations_and_top_features(trained):
    model = trained[0]["logistic_regression"]
    reasons = explain(model, "verify your account urgently at http://x.co")
    assert reasons and all(score > 0 for _, score in reasons)
    feats = top_features(model, n=5)
    assert len(feats["phishing"]) == 5
    assert top_features(trained[0]["naive_bayes"]) == {}
