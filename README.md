# Phishing Email Classifier

![CI](https://github.com/Johnplumb123/phishing-email-classifier/actions/workflows/ci.yml/badge.svg)

A machine learning model that classifies emails as **phishing** or **legitimate** from their text. It compares Logistic Regression with Multinomial Naive Bayes on TF-IDF features, explains which words drove each prediction, and includes a Gradio web demo.

## Why this project

Phishing is still the most common way into an organisation. This project looks at how far a simple, interpretable model can get, and it treats the problem as a security problem rather than just a classification exercise:

- **Recall matters more than accuracy.** A missed phishing email (false negative) can lead to stolen credentials; a false positive just puts a real email in quarantine. The best model is chosen by recall first, then F1.
- **Explainability.** Logistic Regression coefficients show *why* an email was flagged, which a security analyst needs before trusting a verdict.
- **Generalisation.** URLs, email addresses and numbers are replaced with placeholder tokens (`urltoken`, `emailtoken`, `numtoken`). This keeps the signal ("this email contains a link") without the model memorising specific domains attackers will change tomorrow.

## Dataset

Trained on the [Phishing Email Detection dataset on Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails): around 18,650 emails labelled `Safe Email` or `Phishing Email`.

Download `Phishing_Email.csv` into the `data/` folder (it's git-ignored because it's large).

A 50-email toy sample (`data/sample_emails.csv`) is included so tests and CI run without the download. Scores on it are meaningless because it's tiny.

## Results

<!-- Fill this in after training on the full dataset: copy the table printed by train.py -->

| Model | Accuracy | Precision | Recall | F1 | False negatives |
|---|---|---|---|---|---|
| Logistic Regression | – | – | – | – | – |
| Naive Bayes | – | – | – | – | – |

![Confusion matrix](results/confusion_logistic_regression.png)

**Words most associated with phishing:** *(paste from train.py output)*

## Usage

```bash
pip install -r requirements.txt

# 1. Train on the Kaggle dataset
python train.py --data data/Phishing_Email.csv \
    --text-col "Email Text" --label-col "Email Type" --positive-label "Phishing Email"

# 2. Classify an email from the terminal
python predict.py "Your account has been suspended. Verify now at http://secure-login.example"
# PHISHING  (phishing probability: 94.2%)
# Top signals: suspended, verify, urltoken, account

# 3. Web demo
python app.py   # opens http://127.0.0.1:7860
```

`train.py` saves the models, confusion matrix images and a `metrics.json` file to `results/`.

## How it works

1. **Clean**: lowercase the text and swap URLs, emails and numbers for tokens.
2. **Vectorise**: TF-IDF on single words and word pairs (e.g. "verify your" carries more signal than either word alone), with English stopwords removed.
3. **Train**: an 80/20 stratified split, so both sets keep the same phishing/legitimate ratio. Logistic Regression uses `class_weight="balanced"` to handle class imbalance.
4. **Evaluate**: precision, recall, F1 and a confusion matrix for each model.
5. **Explain**: multiply each word's TF-IDF weight by its model coefficient to find which words in *this* email pushed it towards phishing.

## Project structure

```
phishclf/model.py   # cleaning, training, evaluation, explanation
train.py            # train + compare + save results
predict.py          # command-line prediction
app.py              # Gradio web demo
tests/              # pytest suite (runs on the toy sample)
```

## Limitations

- Text only: it doesn't look at headers (SPF/DKIM/DMARC), sender reputation or attachments, which real mail filters rely on heavily.
- Bag-of-words models can be evaded by rephrasing; LLM-written phishing in particular may avoid classic trigger words.
- The dataset is a snapshot, and phishing language changes over time.

**Next steps:** add header-based features, try a fine-tuned transformer (e.g. DistilBERT) and compare it against this baseline, and test on newer, LLM-generated phishing.
