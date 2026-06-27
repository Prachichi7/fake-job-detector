"""
Fake Job Posting Detector — Flask App
Run: python app.py
Then open: http://localhost:5000
"""

import json, os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and metrics
MODEL_PATH   = "model/pipeline.pkl"
METRICS_PATH = "model/metrics.json"

pipeline = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# Red flag keywords that strongly predict fake postings
RED_FLAGS = [
    "no experience needed", "earn from home", "work from home",
    "guaranteed income", "immediate joining", "no interview",
    "urgent hiring", "limited slots", "hurry", "make money fast",
    "send bank details", "western union", "wire transfer",
    "₹80,000", "earn big", "data entry", "online job",
    "part time earn", "whatsapp", "telegram"
]

def analyse_text(text):
    """Predict fraud probability and extract red flags."""
    combined = text.lower()
    proba    = pipeline.predict_proba([combined])[0][1]

    found_flags = [f for f in RED_FLAGS if f in combined]

    # Signal breakdown (heuristics for UI display)
    signals = {
        "urgency_language":  any(w in combined for w in ["urgent", "hurry", "immediate", "limited"]),
        "unrealistic_pay":   any(w in combined for w in ["earn big", "guaranteed", "₹80,000", "earn ₹50,000"]),
        "no_screening":      any(w in combined for w in ["no interview", "no experience", "immediate joining"]),
        "suspicious_contact":any(w in combined for w in ["whatsapp", "telegram", "bank details", "western union"]),
        "poor_company_info": len(combined.split()) < 30,
    }
    signal_score = sum(signals.values())

    return {
        "probability": round(float(proba) * 100, 1),
        "verdict":     "FAKE" if proba >= 0.5 else "REAL",
        "confidence":  "High" if abs(proba - 0.5) > 0.35 else "Medium" if abs(proba - 0.5) > 0.15 else "Low",
        "red_flags":   found_flags,
        "signals":     signals,
        "signal_score":signal_score
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title       = data.get("title", "")
    company     = data.get("company", "")
    description = data.get("description", "")
    requirements= data.get("requirements", "")

    combined = f"{title} {company} {description} {requirements}"

    if len(combined.strip()) < 10:
        return jsonify({"error": "Please enter at least a job title and description."}), 400

    result = analyse_text(combined)
    return jsonify(result)


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
