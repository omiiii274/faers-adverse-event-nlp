# faers-adverse-event-nlp
# 💊 Drug Adverse Event Detection — NLP on FDA FAERS Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![AUC](https://img.shields.io/badge/AUC-0.89-brightgreen)

NLP pipeline to classify adverse event report seriousness using FAERS-style data.

## Results
- **AUC-ROC: 0.89**
- Processed 5,000+ adverse event records
- TF-IDF + Logistic Regression pipeline

## Screenshots
![ROC Curve](images/roc_curve.png)
![Confusion Matrix](images/confusion_matrix.png)
![Events by Drug](images/events_by_drug.png)
![Events Over Time](images/events_by_system_quarter.png)
![Top NLP Features](images/top_nlp_features.png)

## Run
```bash
pip install -r requirements.txt
python main.py
