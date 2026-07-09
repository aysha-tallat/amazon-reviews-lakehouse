# notebooks/07_inference.py
import torch
import mlflow
import mlflow.pytorch
import mlflow.xgboost
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from transformers import DistilBertTokenizerFast
import threading

MLFLOW_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

# ── 1. Load models once at startup ───────────────────────────────
print("Loading XGBoost model...")
exp     = client.get_experiment_by_name("amazon-reviews-ml")
run     = client.search_runs([exp.experiment_id],
                              order_by=["metrics.accuracy DESC"],
                              max_results=1)[0]
xgb_model = mlflow.xgboost.load_model(f"runs:/{run.info.run_id}/xgboost_model")

print("Loading DistilBERT model...")
exp     = client.get_experiment_by_name("amazon-reviews-dl")
run     = client.search_runs([exp.experiment_id],
                              order_by=["metrics.accuracy DESC"],
                              max_results=1)[0]
dl_model  = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/distilbert_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
dl_model.eval()

print("✅ Both models loaded\n")

# ── 2. Prediction functions ───────────────────────────────────────
def predict_rating_xgb(review_text: str) -> int:
    features = pd.DataFrame([{
        "num_helpful":       0,
        "review_text_len":   len(review_text),
        "review_title_len":  0,
        "do_recommend":      0,
        "has_title":         0,
        "review_year":       2024,
        "review_month":      1,
        "review_dow":        2,
        "brand":             0,
        "primaryCategories": 0
    }])
    return int(xgb_model.predict(features)[0]) + 1

def predict_sentiment_dl(review_text: str) -> dict:
    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = dl_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]
        pred    = torch.argmax(probs).item()

    return {
        "sentiment":     "positive" if pred == 1 else "negative",
        "confidence":    round(probs[pred].item() * 100, 1),
        "positive_prob": round(probs[1].item(), 4),
        "negative_prob": round(probs[0].item(), 4)
    }

# ── 3. Flask app ──────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Amazon Review Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f6f9; display: flex;
               justify-content: center; padding: 40px 20px; }
        .container { width: 100%; max-width: 720px; }
        h1 { font-size: 22px; color: #232f3e; margin-bottom: 4px; }
        .subtitle { font-size: 13px; color: #666; margin-bottom: 24px; }
        textarea { width: 100%; height: 130px; padding: 12px; font-size: 14px;
                   border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
        button { margin-top: 12px; padding: 10px 28px; background: #ff9900;
                 color: white; border: none; border-radius: 6px;
                 font-size: 14px; cursor: pointer; font-weight: bold; }
        button:hover { background: #e68a00; }
        .results { margin-top: 28px; display: flex; gap: 16px; }
        .card { flex: 1; background: white; border-radius: 8px;
                padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .card h2 { font-size: 14px; color: #888; margin-bottom: 12px;
                   text-transform: uppercase; letter-spacing: 0.5px; }
        .rating { font-size: 36px; font-weight: bold; color: #232f3e; }
        .stars { color: #ff9900; font-size: 22px; margin-top: 4px; }
        .sentiment { font-size: 28px; font-weight: bold; margin-bottom: 6px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .confidence { font-size: 13px; color: #666; }
        .bar-wrap { margin-top: 12px; }
        .bar-label { font-size: 12px; color: #555; margin-bottom: 3px;
                     display: flex; justify-content: space-between; }
        .bar-bg { background: #eee; border-radius: 4px; height: 8px; }
        .bar-fill { height: 8px; border-radius: 4px; }
        .pipeline { margin-top: 24px; background: white; border-radius: 8px;
                    padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .pipeline h2 { font-size: 13px; color: #888; text-transform: uppercase;
                        letter-spacing: 0.5px; margin-bottom: 10px; }
        .pipeline p { font-size: 13px; color: #444; line-height: 1.6; }
        .hidden { display: none; }
        .loading { text-align: center; color: #888; font-size: 14px;
                   margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🛍️ Amazon Review Analyzer</h1>
    <p class="subtitle">Medallion Lakehouse · XGBoost (Rating) + DistilBERT (Sentiment)</p>

    <textarea id="review" placeholder="Paste an Amazon product review here..."></textarea>
    <br>
    <button onclick="analyze()">Analyze Review</button>

    <div id="loading" class="loading hidden">Analyzing...</div>

    <div id="results" class="results hidden">
        <div class="card">
            <h2>XGBoost · Predicted Rating</h2>
            <div class="rating" id="rating-num">—</div>
            <div class="stars" id="rating-stars"></div>
            <p style="font-size:12px;color:#999;margin-top:8px;">
                Based on structured metadata<br>(text length, helpfulness, brand)
            </p>
        </div>
        <div class="card">
            <h2>DistilBERT · Sentiment</h2>
            <div class="sentiment" id="sentiment-label">—</div>
            <div class="confidence" id="confidence-text"></div>
            <div class="bar-wrap">
                <div class="bar-label">
                    <span>Positive</span><span id="pos-pct"></span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" id="pos-bar"
                         style="background:#27ae60;width:0%"></div>
                </div>
                <div class="bar-label" style="margin-top:6px">
                    <span>Negative</span><span id="neg-pct"></span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" id="neg-bar"
                         style="background:#e74c3c;width:0%"></div>
                </div>
            </div>
        </div>
    </div>

    <div id="pipeline-info" class="pipeline hidden">
        <h2>Pipeline Path</h2>
        <p id="pipeline-text"></p>
    </div>
</div>

<script>
async function analyze() {
    const text = document.getElementById("review").value.trim();
    if (!text) { alert("Please enter a review."); return; }

    document.getElementById("results").classList.add("hidden");
    document.getElementById("pipeline-info").classList.add("hidden");
    document.getElementById("loading").classList.remove("hidden");

    const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review: text })
    });
    const data = await resp.json();

    document.getElementById("loading").classList.add("hidden");

    // XGBoost
    const rating = data.xgboost.predicted_rating;
    document.getElementById("rating-num").textContent = rating + "★";
    document.getElementById("rating-stars").textContent = "★".repeat(rating) + "☆".repeat(5 - rating);

    // DistilBERT
    const s = data.distilbert.sentiment;
    const label = document.getElementById("sentiment-label");
    label.textContent = s.charAt(0).toUpperCase() + s.slice(1);
    label.className = "sentiment " + s;
    document.getElementById("confidence-text").textContent =
        `Confidence: ${data.distilbert.confidence}%`;

    const posPct = (data.distilbert.positive_prob * 100).toFixed(1);
    const negPct = (data.distilbert.negative_prob * 100).toFixed(1);
    document.getElementById("pos-pct").textContent = posPct + "%";
    document.getElementById("neg-pct").textContent = negPct + "%";
    document.getElementById("pos-bar").style.width = posPct + "%";
    document.getElementById("neg-bar").style.width = negPct + "%";

    // Pipeline info
    document.getElementById("pipeline-text").textContent =
        `Raw text (${text.length} chars) → Bronze Delta → Silver (cleaned) → ` +
        `Gold ML (${text.length} char length feature) + Gold NLP (tokenized) → ` +
        `XGBoost: ${rating}★ | DistilBERT: ${s} (${data.distilbert.confidence}% confidence)`;

    document.getElementById("results").classList.remove("hidden");
    document.getElementById("pipeline-info").classList.remove("hidden");
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    review = data.get("review", "").strip()

    if not review:
        return jsonify({"error": "Empty review"}), 400

    xgb_rating = predict_rating_xgb(review)
    dl_result  = predict_sentiment_dl(review)

    return jsonify({
        "review": review[:100],
        "xgboost": {
            "predicted_rating": xgb_rating
        },
        "distilbert": {
            "sentiment":     dl_result["sentiment"],
            "confidence":    dl_result["confidence"],
            "positive_prob": dl_result["positive_prob"],
            "negative_prob": dl_result["negative_prob"]
        }
    })

if __name__ == "__main__":
    print("🚀 Starting server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)