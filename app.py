"""Web demo: paste an email, get a verdict. Run `python app.py` after training."""
import gradio as gr
import joblib

from phishclf.model import explain, predict

model = joblib.load("results/model_lr.joblib")


def classify(text: str):
    if not text.strip():
        return "Paste an email first.", {}, ""
    verdict, prob = predict(model, text)
    reasons = explain(model, text, n=8)
    signals = ", ".join(w for w, _ in reasons) or "No strong phishing signals"
    return verdict, {"Phishing": prob, "Legitimate": 1 - prob}, signals


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=12, label="Email text"),
    outputs=[gr.Textbox(label="Verdict"), gr.Label(label="Confidence"),
             gr.Textbox(label="Words that raised suspicion")],
    title="Phishing Email Classifier",
    description="TF-IDF + Logistic Regression. A demo, not a replacement for a real mail filter.",
    examples=[
        ["Dear customer, your account has been suspended due to unusual activity. "
         "Verify your identity within 24 hours at http://secure-login.example to avoid closure."],
        ["Hi team, attached are the notes from Tuesday's sprint review. "
         "Let me know if I missed anything before Friday."],
    ],
)

if __name__ == "__main__":
    demo.launch()
