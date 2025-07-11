import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier
import warnings
import joblib

warnings.filterwarnings("ignore")

# === Directories ===
os.makedirs('xgboost_output', exist_ok=True)

# === Load normalized data ===
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)
X_raw = data['data']
y = np.array(data['labels'])

print("Original X_raw type:", type(X_raw))
print("Sample feature type:", type(X_raw[0]))
print("Sample feature length:", len(X_raw[0]))

# Convert X_raw to numpy array of floats ensuring fixed length per sample
X = np.array([np.array(xi, dtype=float) for xi in X_raw])
print("X shape after conversion:", X.shape)

# === Encode labels ===
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# === Train XGBoost ===
model = XGBClassifier(eval_metric='mlogloss', verbosity=0, use_label_encoder=False)
model.fit(X_train, y_train)

# === Save model ===
joblib.dump({'model': model, 'label_encoder': le}, 'xgboost_output/model.pkl')

# === Predictions & accuracy ===
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# === 1. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig('xgboost_output/confusion_matrix.png', dpi=200)
plt.close()

# === 2. Classification Report ===
rep = classification_report(y_test, y_pred, target_names=classes, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(rep).iloc[:-1, :-1].T
plt.figure(figsize=(10, 6))
sns.heatmap(rep_df, annot=True, cmap='YlGnBu')
plt.title("Classification Report")
plt.savefig('xgboost_output/classification_report.png', dpi=200)
plt.close()

# === 3. ROC Curves ===
n = len(classes)
y_bin = label_binarize(y_test, classes=range(n))
plt.figure(figsize=(10, 8))
for i in range(n):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
    score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC (One-vs-Rest)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(fontsize='small', ncol=2)
plt.savefig('xgboost_output/roc_curve.png', dpi=200)
plt.close()

# === 4. Precision–Recall Curves ===
plt.figure(figsize=(10, 8))
for i in range(n):
    pr, rc, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
    plt.plot(rc, pr, label=f'{classes[i]}')
plt.title("Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(fontsize='small', ncol=2)
plt.savefig('xgboost_output/precision_recall_curve.png', dpi=200)
plt.close()

# === 5. Learning Curve ===
train_sizes = [0.2, 0.5, 0.8]
train_sizes_abs = [int(p * len(X)) for p in train_sizes]
tc, ts = learning_curve(
    model, X, y_enc, cv=3, scoring='accuracy',
    train_sizes=train_sizes, n_jobs=-1
)[1:3]  # train_scores, test_scores

tc_mean = np.mean(tc, axis=1) if tc.ndim == 2 else tc
ts_mean = np.mean(ts, axis=1) if ts.ndim == 2 else ts

plt.figure(figsize=(8, 6))
plt.plot(train_sizes_abs, tc_mean, '-o', label='Train')
plt.plot(train_sizes_abs, ts_mean, '-o', label='Validation')
plt.title("Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig('xgboost_output/learning_curve.png', dpi=200)
plt.close()

# === 6. Feature Importance ===
plt.figure(figsize=(10, 6))
imp = model.feature_importances_
plt.bar(range(len(imp)), imp)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.savefig('xgboost_output/feature_importance.png', dpi=200)
plt.close()

# === 7. SHAP Summary ===
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_train[:100])
shap.summary_plot(sv, X_train[:100], show=False)
plt.savefig('xgboost_output/shap_summary.png', dpi=200)
plt.close()
