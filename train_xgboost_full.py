import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from collections import Counter

# === Setup folders ===
base_vis_dir = "visualizations"
xgb_vis_dir = os.path.join(base_vis_dir, "xgboost")
os.makedirs(xgb_vis_dir, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(xgb_vis_dir, filename))
    plt.close(fig)

# === Load Data ===
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

X_raw = data['data']
y_raw = np.array(data['labels'])

# === Preprocess: pad + flatten ===
max_len = max(len(np.array(s).flatten()) for s in X_raw)
X = np.array([
    np.pad(np.array(s).flatten(), (0, max_len - len(np.array(s).flatten())), mode='constant')
    if len(np.array(s).flatten()) < max_len else np.array(s).flatten()[:max_len]
    for s in X_raw
])

# === Encode labels to integers ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = label_encoder.classes_

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train XGBoost
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# === Predict and decode labels
y_pred = model.predict(X_test)
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# === Accuracy and Classification Report
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

# === Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
save_plot(fig, "xgb_confusion_matrix.png")

# === ROC + PR (only for binary)
if len(class_names) == 2:
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=1)
    auc = roc_auc_score(y_test, y_probs)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    save_plot(fig, "xgb_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, y_probs, pos_label=1)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    save_plot(fig, "xgb_precision_recall_curve.png")

# === Feature Importance
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
top_indices = np.argsort(model.feature_importances_)[::-1][:30]
top_importances = model.feature_importances_[top_indices]
sns.barplot(x=top_indices, y=top_importances, ax=ax)
ax.set_title("Top 30 XGBoost Feature Importances")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Importance")
save_plot(fig, "xgb_feature_importance.png")

# === SHAP
try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], plot_type="bar", show=False)
    plt.savefig(os.path.join(xgb_vis_dir, "xgb_shap_summary_bar.png"))
    plt.close()
    shap.summary_plot(shap_values, X_test[:100], show=False)
    plt.savefig(os.path.join(xgb_vis_dir, "xgb_shap_summary_dot.png"))
    plt.close()
except Exception as e:
    print("SHAP Error:", e)

# === t-SNE
try:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_test)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_test_decoded, palette='bright', ax=ax)
    ax.set_title("t-SNE: Test Data Class Separation")
    save_plot(fig, "xgb_tsne.png")
except Exception as e:
    print("t-SNE Error:", e)

# === Error Analysis
errors = y_test != y_pred
error_counts = Counter(label_encoder.inverse_transform(y_test[errors]))
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.bar(error_counts.keys(), error_counts.values())
ax.set_title("Errors per Class")
ax.set_xlabel("Actual Class")
ax.set_ylabel("Error Count")
save_plot(fig, "xgb_error_analysis.png")

# === Save Model
os.makedirs("models", exist_ok=True)
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ All visualizations saved in: {xgb_vis_dir}")
