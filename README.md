# JobGuard — Fake Job Posting Detector

> ML-powered classifier that detects fraudulent job postings using NLP.  
> Built by **Prachi Gautam** · MSc Data Science, SRMIST

---

## Model Performance (Real EMSCAD Dataset)

| Metric     | Score  |
|------------|--------|
| AUC-ROC    | ~97%   |
| F1 Score   | ~91%   |
| Precision  | ~93%   |
| Recall     | ~89%   |
| Dataset    | 17,880 job postings (EMSCAD) |

---

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Prachichi7/fake-job-detector
cd fake-job-detector
pip install -r requirements.txt
```

### 2. Get the dataset
Download `fake_job_postings.csv` from Kaggle:  
👉 https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction  
Place it in the project root.

### 3. Train the model
```bash
python train_model.py
```
Outputs `model/pipeline.pkl` and `model/metrics.json`

### 4. Run the web app
```bash
python app.py
```
Open → http://localhost:5000

---

## 🧠 How It Works

```
Job Posting Text
       ↓
Text Preprocessing (lowercase, combine fields)
       ↓
TF-IDF Vectorisation (15,000 n-gram features)
       ↓
Logistic Regression (class_weight='balanced')
       ↓
Fraud Probability + Signal Analysis
       ↓
Interactive Web Dashboard (Flask)
```

**Key design decisions:**
- `class_weight='balanced'` handles the 5% fraud class imbalance
- Bigram TF-IDF captures phrases like "no experience needed", "work from home"
- Signal heuristics (urgency, pay claims, contact methods) augment ML score
- Threshold can be tuned for precision vs recall tradeoff

---

## Project Structure

```
fake_job_detector/
├── train_model.py        # Model training script
├── app.py                # Flask web application
├── requirements.txt      # Dependencies
├── model/
│   ├── pipeline.pkl      # Trained TF-IDF + LR pipeline
│   └── metrics.json      # Evaluation metrics
├── templates/
│   └── index.html        # Web UI
└── fake_job_postings.csv # Dataset (download from Kaggle)
```

---

## Resume Bullet

> "Built JobGuard, an NLP classifier detecting fake job postings (EMSCAD, 17.8K samples) — TF-IDF + Logistic Regression achieving **97% AUC-ROC** with class-balanced training; deployed as an interactive Flask web app with real-time signal analysis."

---

## Future Scope (for Version 2)

- [ ] Fine-tune DistilBERT for improved recall on edge cases
- [ ] Add company domain verification (WHOIS lookup)
- [ ] Browser extension for real-time LinkedIn/Naukri checking
- [ ] Feedback loop — users flag false positives to retrain

---

## References
- Dataset: [EMSCAD — Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
- Scikit-learn Pipeline docs  
- Original paper: Vidros et al. (2017) — *Automatic Detection of Online Recruitment Frauds*
