
---

### PROJECT 2: `faers-adverse-event-nlp`

**File: `main.py`**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os, re, warnings
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)
np.random.seed(42)

# ── Generate Realistic FAERS-style Data ──
print("📊 Generating simulated FAERS adverse event data...")

drugs = ['Aspirin','Ibuprofen','Metformin','Lisinopril','Atorvastatin',
         'Omeprazole','Amoxicillin','Metoprolol','Losartan','Gabapentin']

serious_symptoms = [
    'patient experienced severe anaphylactic shock requiring emergency hospitalization',
    'reported acute liver failure with elevated enzymes and jaundice symptoms',
    'cardiac arrest occurred during treatment requiring ICU admission',
    'severe Stevens-Johnson syndrome with extensive skin involvement noted',
    'acute kidney injury with elevated creatinine and oliguria observed',
    'patient developed pulmonary embolism with dyspnea and chest pain',
    'severe thrombocytopenia leading to spontaneous bleeding episodes',
    'reported seizures and loss of consciousness during medication use',
    'acute pancreatitis with severe abdominal pain and lipase elevation',
    'patient experienced rhabdomyolysis with elevated CK levels'
]

non_serious_symptoms = [
    'patient reported mild headache and nausea after taking medication',
    'experienced mild dizziness that resolved without intervention',
    'reported mild gastrointestinal discomfort including bloating',
    'patient noted mild skin rash on forearms that was self-limiting',
    'mild fatigue and drowsiness reported during first week of treatment',
    'patient experienced mild dry mouth and decreased appetite',
    'reported mild joint pain and muscle stiffness in morning',
    'experienced occasional mild insomnia during treatment period',
    'patient noted mild constipation managed with dietary changes',
    'reported mild anxiety and restlessness during initial dosing'
]

body_systems = ['Cardiac','Hepatic','Renal','Dermatologic','Neurologic',
                'Gastrointestinal','Hematologic','Respiratory','Musculoskeletal','Psychiatric']

n = 5000
narratives = []
labels = []
drug_list = []
system_list = []
quarters = []

for i in range(n):
    if np.random.random() < 0.35:
        narrative = np.random.choice(serious_symptoms)
        noise = np.random.choice([' patient is elderly.', ' history of comorbidities.',
                                   ' required prolonged monitoring.', ' dose was adjusted.', ''])
        narratives.append(narrative + noise)
        labels.append(1)
    else:
        narrative = np.random.choice(non_serious_symptoms)
        noise = np.random.choice([' no treatment needed.', ' resolved in 2 days.',
                                   ' patient continued therapy.', ' no dose change.', ''])
        narratives.append(narrative + noise)
        labels.append(0)
    
    drug_list.append(np.random.choice(drugs))
    system_list.append(np.random.choice(body_systems))
    quarters.append(np.random.choice(['2023-Q1','2023-Q2','2023-Q3','2023-Q4',
                                       '2024-Q1','2024-Q2','2024-Q3','2024-Q4']))

df = pd.DataFrame({
    'narrative': narratives, 'serious': labels, 'drug': drug_list,
    'body_system': system_list, 'quarter': quarters
})
df.to_csv('faers_simulated_data.csv', index=False)
print(f"✅ Generated {len(df)} records | Serious: {sum(labels)} | Non-serious: {n - sum(labels)}")

# ── NLP Pipeline ──
print("\n🔬 Building NLP classification pipeline...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df['narrative'])
y = df['serious']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

print(f"\n📈 Results:")
print(f"AUC-ROC: {auc_score:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ── Plot 1: ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#2B7A78', lw=2, label=f'AUC = {auc_score:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Adverse Event Seriousness Classification', fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curve.png', dpi=150)
print("✅ Saved: images/roc_curve.png")

# ── Plot 2: Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Serious', 'Serious'])
ax.set_yticklabels(['Non-Serious', 'Serious'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18,
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150)
print("✅ Saved: images/confusion_matrix.png")

# ── Plot 3: Adverse Events by Drug ──
drug_serious = df.groupby('drug')['serious'].mean().sort_values(ascending=True) * 100
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(drug_serious.index, drug_serious.values, color='#2B7A78')
ax.set_xlabel('Serious Event Rate (%)')
ax.set_title('Serious Adverse Event Rate by Drug')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, drug_serious.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')
plt.tight_layout()
plt.savefig('images/events_by_drug.png', dpi=150)
print("✅ Saved: images/events_by_drug.png")

# ── Plot 4: Events by Body System and Quarter ──
pivot = df.groupby(['quarter', 'body_system'])['serious'].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
for system in df['body_system'].unique():
    subset = pivot[pivot['body_system'] == system]
    ax.plot(subset['quarter'], subset['serious'], marker='o', label=system)
ax.set_xlabel('Quarter'); ax.set_ylabel('Number of Serious Events')
ax.set_title('Serious Adverse Events by Body System Over Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/events_by_system_quarter.png', dpi=150)
print("✅ Saved: images/events_by_system_quarter.png")

# ── Plot 5: Top NLP Features ──
feature_names = tfidf.get_feature_names_out()
coefs = model.coef_[0]
top_positive = pd.Series(coefs, index=feature_names).nlargest(15)
top_negative = pd.Series(coefs, index=feature_names).nsmallest(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(top_positive.index, top_positive.values, color='#E74C3C')
axes[0].set_title('Top Words → Serious')
axes[0].grid(axis='x', alpha=0.3)
axes[1].barh(top_negative.index, top_negative.values, color='#2B7A78')
axes[1].set_title('Top Words → Non-Serious')
axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('images/top_nlp_features.png', dpi=150)
print("✅ Saved: images/top_nlp_features.png")

print(f"\n🎯 FINAL AUC: {auc_score:.4f}")
print("🏁 All outputs saved to /images/ folder")
