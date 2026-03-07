# 💊 Drug Adverse Event Detection — NLP on FDA FAERS Data

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.89-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

NLP pipeline to classify adverse event report seriousness from FDA FAERS data. Achieved **AUC 0.89**.

---

## 📊 Results

### ROC Curve
![ROC Curve](images/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Serious Event Rate by Drug
![Events by Drug](images/events_by_drug.png)

### Serious Events — Quarterly Trend
![Quarterly](images/events_quarterly.png)

### Body System × Drug Heatmap
![Heatmap](images/heatmap.png)

### NLP Feature Importance
![NLP Features](images/nlp_features.png)

---

## 🛠️ Tech Stack
| Category | Tools |
|----------|-------|
| NLP | TF-IDF, Logistic Regression, scikit-learn |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib |

## 🚀 How to Run
```bash
git clone https://github.com/omiiii274/faers-adverse-event-nlp.git
cd faers-adverse-event-nlp
pip install -r requirements.txt
python build.py
