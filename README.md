
# 💊 Drug Adverse Event Detection — NLP on FDA FAERS Data

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.89-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## 📋 About This Project

This project uses Natural Language Processing (NLP) to automatically read adverse event reports and determine whether a reported drug side effect is **serious** (patient needed hospital, life-threatening, caused disability) or **non-serious** (mild headache, temporary nausea, minor rash).

The data simulates what pharmaceutical companies receive from the FDA Adverse Event Reporting System (FAERS) — a real database where doctors, pharmacists, and patients report drug side effects. The FDA receives over 2 million reports per year, and each one contains a free-text narrative describing what happened to the patient.

## ❓ Problem This Solves

Pharmaceutical companies are legally required to monitor drug safety after a product reaches the market. Their pharmacovigilance teams must:

1. Read every adverse event report
2. Decide if it is serious or non-serious
3. Flag potential safety signals
4. Report findings to regulators (FDA, EMA)

With millions of reports, manual review is:
- **Slow** — thousands of reports per drug per quarter
- **Inconsistent** — different reviewers may classify the same report differently
- **Expensive** — requires large teams of medical reviewers

An NLP model can provide a fast, consistent first-pass classification, letting human reviewers focus on the most critical cases.

## 🔬 How It Works

**Step 1: Data**
- Created 5,000 simulated adverse event reports
- Each report has a free-text narrative, the drug name, body system affected, and quarter reported
- 35% of reports are serious, 65% are non-serious
- Covers 10 different drugs and 10 body systems over 12 quarters (2022-2024)

**Step 2: Text Processing (NLP)**
- Converted all narrative text to numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
- TF-IDF measures how important each word is in a report compared to all other reports
- Used both single words and word pairs (bigrams) — so "liver failure" is captured as one feature, not two separate words
- Maximum 3,000 features, English stop words removed

**Step 3: Model Training**
- Split data: 80% training, 20% testing (stratified to keep same ratio of serious/non-serious)
- Trained Logistic Regression classifier
- Model learns which words are associated with serious events (e.g., "hospitalization", "failure", "emergency") vs non-serious events (e.g., "mild", "resolved", "temporary")

**Step 4: Evaluation**
- AUC-ROC: 0.89 (very good — 1.0 would be perfect, 0.5 would be random guessing)
- Precision for serious events: 0.87 (when the model says "serious", it is correct 87% of the time)
- Recall for serious events: 0.85 (the model catches 85% of all actual serious events)

**Step 5: Safety Signal Analysis**
- Generated dashboards showing which drugs have the highest serious event rates
- Created body system × drug heatmap to identify specific organ toxicity patterns
- Tracked quarterly trends to detect emerging safety signals over time

## 📊 Key Numbers

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.89** |
| Precision (Serious) | 0.87 |
| Recall (Serious) | 0.85 |
| F1-Score (Serious) | 0.86 |
| Total reports processed | 5,000 |
| Drugs analysed | 10 |
| Body systems covered | 10 |
| Time period | 12 quarters (2022-2024) |

## 🛠️ Tools Used

| What | Tool |
|------|------|
| NLP technique | TF-IDF Vectorisation with bigrams |
| ML model | Logistic Regression |
| ML framework | scikit-learn |
| Data handling | Pandas, NumPy |
| Plotting | Matplotlib |
| Evaluation | ROC-AUC, Precision, Recall, F1, Confusion Matrix |

## 📁 Files In This Project
