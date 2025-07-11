import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from collections import Counter

# === Directory Setup ===
base_vis_dir = "visualizations"
vis_dir = os.path.join(base_vis_dir, "cnn_rnn")
os.makedirs(vis_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

def save_plot(fig, name):
    fig.savefig(os.path.join(vis_dir, name), bbox_inches="tight")
    plt.close(fig)

# === Load Data ===
with open("data.pickle", "rb") as f:
    data = pickle.load(f)

X_raw = data["data"]
y_raw = np.array(data["labels"])

max_len = max(len(np.array(seq).flatten()) for seq in X_raw)
X = np.array([
    np.pad(np.array(seq).flatten(), (0, max_len - len(np.array(seq).flatten())))
    for seq in X_raw
])
X = X[..., np.newaxis]  # (samples, time_steps, 1)

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)
class_names = label_encoder.classes_

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Define CNN + LSTM Model ===
input_layer = Input(shape=(X.shape[1], 1))
x = Conv1D(64, kernel_size=3, activation="relu")(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = LSTM(128)(x)
x = Dropout(0.3)(x)
output_layer = Dense(y_categorical.shape[1], activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# === Train Model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# === Save Model ===
model.save("models/cnn_rnn_model.h5")

# === Predictions ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# === Confusion Matrix ===
fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("CNN + RNN Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
save_plot(fig, "confusion_matrix.png")

# === Classification Report ===
print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels, zero_division=0))

# === Accuracy and Loss Curves ===
fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title("Accuracy Curve")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[0].legend()

ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title("Loss Curve")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend()
save_plot(fig, "learning_curves.png")

# === t-SNE Plot ===
try:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_test.reshape(X_test.shape[0], -1))
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true_labels, palette="tab10", ax=ax)
    ax.set_title("t-SNE of Test Set")
    ax.legend(loc="best", fontsize="small")
    save_plot(fig, "tsne.png")
except Exception as e:
    print("t-SNE Error:", e)

# === Error Analysis Plot ===
errors = y_true != y_pred
error_counts = Counter(label_encoder.inverse_transform(y_true[errors]))
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
ax.bar(error_counts.keys(), error_counts.values())
ax.set_title("Misclassifications per Class")
ax.set_xlabel("True Class")
ax.set_ylabel("Error Count")
save_plot(fig, "error_analysis.png")

# === Saliency Map (1D Grad-CAM style) ===
sample_idx = 0
sample_input = tf.convert_to_tensor(np.expand_dims(X_test[sample_idx], axis=0), dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(sample_input)
    preds = model(sample_input)
    loss = preds[:, tf.argmax(preds[0])]

grads = tape.gradient(loss, sample_input)[0].numpy().squeeze()
weights = np.abs(grads)
weights /= np.max(weights)

fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
ax.plot(X_test[sample_idx].squeeze(), label='Input Signal')
ax.imshow(np.expand_dims(weights, axis=0), cmap='jet', alpha=0.6,
          extent=(0, X.shape[1], ax.get_ylim()[0], ax.get_ylim()[1]))
ax.set_title(f"Saliency Map for class: {label_encoder.inverse_transform([y_pred[sample_idx]])[0]}")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Amplitude")
save_plot(fig, "saliency_map.png")

print(f"\n✅ CNN + RNN model trained and visualizations saved in: {vis_dir}")
