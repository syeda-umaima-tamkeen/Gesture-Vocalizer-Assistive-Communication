import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from collections import Counter

# Create directory for saving visualizations inside svm folder
base_vis_dir = "visualizations"
svm_vis_dir = os.path.join(base_vis_dir, "svm")
os.makedirs(svm_vis_dir, exist_ok=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(svm_vis_dir, filename))
    plt.close(fig)  # Close plot to avoid display

# Load dataset
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

print("Keys found in data.pickle:", list(data.keys()))

X_raw = data['data']
y = np.array(data['labels'])

print(f"Number of samples: {len(X_raw)}")

max_len = max(len(sample.flatten()) if hasattr(sample, "flatten") else len(sample) for sample in X_raw)
print(f"Maximum sample length after flattening: {max_len}")

X_processed = []

for sample in X_raw:
    sample = np.array(sample)
    flat_sample = sample.flatten()
    if len(flat_sample) < max_len:
        pad_width = max_len - len(flat_sample)
        flat_sample = np.pad(flat_sample, (0, pad_width), mode='constant')
    elif len(flat_sample) > max_len:
        flat_sample = flat_sample[:max_len]
    X_processed.append(flat_sample)

X = np.array(X_processed)
print(f"Processed feature matrix shape: {X.shape}")

unique_labels = np.unique(y)
label_names = {label: label for label in unique_labels}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ SVM Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))  # <-- fix here

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
fig.tight_layout()
save_plot(fig, "svm_confusion_matrix.png")

if len(unique_labels) == 2:
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=unique_labels[1])
    auc = roc_auc_score(y_test, y_probs)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "svm_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, y_probs, pos_label=unique_labels[1])
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    save_plot(fig, "svm_precision_recall_curve.png")

try:
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=np.argsort(importances)[::-1], y=np.sort(importances)[::-1], ax=ax)
    ax.set_title("Feature Importances (Permutation Importance)")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    save_plot(fig, "svm_permutation_feature_importance.png")
except Exception as e:
    print(f"Permutation importance error: {e}")

try:
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(shap.sample(X_test, 100), nsamples=100)
    class_to_plot = 1 if len(unique_labels) == 2 else 0
    shap.summary_plot(shap_values[class_to_plot], shap.sample(X_test, 100), plot_type="bar", show=False)
    plt.savefig(os.path.join(svm_vis_dir, "svm_shap_summary_bar.png"))
    plt.close()
    shap.summary_plot(shap_values[class_to_plot], shap.sample(X_test, 100), show=False)
    plt.savefig(os.path.join(svm_vis_dir, "svm_shap_summary_dot.png"))
    plt.close()
except Exception as e:
    print(f"SHAP error: {e}")

try:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_test)
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_test, palette='bright', ax=ax)
    ax.set_title("t-SNE: Test Data")
    fig.tight_layout()
    save_plot(fig, "svm_tsne.png")
except Exception as e:
    print(f"t-SNE error: {e}")

errors = y_test != y_pred
error_counts = Counter(y_test[errors])
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(error_counts.keys(), error_counts.values())
ax.set_title("Errors per Class")
ax.set_xlabel("Actual Class")
ax.set_ylabel("Error Count")
fig.tight_layout()
save_plot(fig, "svm_error_analysis.png")

os.makedirs("models", exist_ok=True)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"✅ All visualizations saved inside folder: {svm_vis_dir}")
