"""
Fake Job Posting Detector — Model Training
Dataset: EMSCAD (https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
Download fake_job_postings.csv from Kaggle and place it in the same folder.

Run: python train_model.py
Outputs: model/pipeline.pkl and model/metrics.json
"""

import pandas as pd
import numpy as np
import json, os, joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder

# ─── 1. Load Data ────────────────────────────────────────────────────────────

CSV_PATH = "fake_job_postings.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded real dataset: {df.shape}")
else:
    # Synthetic fallback that mirrors EMSCAD structure (for demo/testing)
    print("⚠️  fake_job_postings.csv not found — generating synthetic data for demo.")
    print("   Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n")
    np.random.seed(42)
    n = 1000

    real_titles   = ["Software Engineer", "Data Analyst", "Marketing Manager", "HR Executive",
                     "Product Manager", "Business Analyst", "Sales Executive", "DevOps Engineer"]
    fake_titles   = ["Earn ₹50,000 From Home!", "URGENT HIRING — No Experience", "Work From Home Agent",
                     "Make Money Fast", "Online Data Entry — Immediate Joining"]
    real_descs    = ["We are looking for a skilled engineer to join our team. You will work on scalable products.",
                     "Exciting opportunity at a fast-growing startup. Strong Python and SQL skills required.",
                     "Join our dynamic marketing team. Experience with digital campaigns preferred."]
    fake_descs    = ["NO EXPERIENCE NEEDED!! Earn BIG from home. Send resume immediately to claim your spot.",
                     "GUARANTEED income. Work 2 hours per day and earn ₹80,000 monthly. Hurry — limited slots!",
                     "Immediate joining. No interview required. Send your bank details to proceed."]

    is_fraud = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    titles   = [np.random.choice(fake_titles) if f else np.random.choice(real_titles)  for f in is_fraud]
    descs    = [np.random.choice(fake_descs)  if f else np.random.choice(real_descs)   for f in is_fraud]
    salaries = [""  if f else np.random.choice(["₹8-12 LPA", "₹5-8 LPA", "$70,000-$90,000", ""]) for f in is_fraud]
    company  = ["" if f else np.random.choice(["Infosys", "TCS", "Google", "Wipro", "Startup Inc"]) for f in is_fraud]

    df = pd.DataFrame({
        "title": titles, "company_profile": company,
        "description": descs, "requirements": descs,
        "salary_range": salaries, "fraudulent": is_fraud
    })
    print(f"   Synthetic dataset: {df.shape}  |  Fraud rate: {is_fraud.mean():.1%}\n")

# ─── 2. Feature Engineering ───────────────────────────────────────────────────

TEXT_COLS = ["title", "company_profile", "description", "requirements"]
for col in TEXT_COLS:
    df[col] = df[col].fillna("")

# Combine text fields into one rich feature
df["combined_text"] = (
    df["title"].str.lower() + " " +
    df["company_profile"].str.lower() + " " +
    df["description"].str.lower() + " " +
    df["requirements"].str.lower()
)

# Extra signal features
df["has_salary"]         = df["salary_range"].apply(lambda x: 1 if str(x).strip() else 0)
df["has_company"]        = df["company_profile"].apply(lambda x: 1 if str(x).strip() else 0)
df["desc_len"]           = df["description"].apply(len)
df["exclamation_count"]  = df["combined_text"].apply(lambda x: x.count("!"))
df["caps_ratio"]         = df["combined_text"].apply(
                               lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

print("Feature columns added. Class balance:")
print(df["fraudulent"].value_counts())
print()

# ─── 3. Train / Test Split ────────────────────────────────────────────────────

X_text  = df["combined_text"]
y       = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y)

# ─── 4. TF-IDF + Logistic Regression Pipeline ────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        class_weight="balanced",   # handles class imbalance (5% fraud)
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    ))
])

pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ─── 5. Metrics ───────────────────────────────────────────────────────────────

auc     = roc_auc_score(y_test, y_proba)
f1      = f1_score(y_test, y_pred, zero_division=0)
prec    = precision_score(y_test, y_pred, zero_division=0)
rec     = recall_score(y_test, y_pred, zero_division=0)
cm      = confusion_matrix(y_test, y_pred).tolist()
cv      = cross_val_score(pipeline, X_text, y, cv=5, scoring="roc_auc")

print("=" * 50)
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  CV AUC    : {cv.mean():.4f} ± {cv.std():.4f}")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ─── 6. Save Artefacts ────────────────────────────────────────────────────────

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/pipeline.pkl")

metrics = {
    "auc_roc":   round(auc,  4),
    "f1_score":  round(f1,   4),
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "cv_auc_mean": round(float(cv.mean()), 4),
    "cv_auc_std":  round(float(cv.std()),  4),
    "confusion_matrix": cm,
    "train_size": int(len(X_train)),
    "test_size":  int(len(X_test)),
    "total_samples": int(len(df)),
    "fraud_rate": round(float(y.mean()), 4)
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved: model/pipeline.pkl  |  model/metrics.json")
print("   Next: python app.py")
