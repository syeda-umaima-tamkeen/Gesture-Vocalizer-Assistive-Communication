import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from collections import Counter

# --- Setup directories ---
base_vis_dir = "visualizations"
cnn_vis_dir = os.path.join(base_vis_dir, "cnn")
os.makedirs(cnn_vis_dir, exist_ok=True)

def save_plot(fig, name):
    fig.savefig(os.path.join(cnn_vis_dir, name), bbox_inches='tight')
    plt.close(fig)

# --- Load data ---
with open("data.pickle", "rb") as f:
    data = pickle.load(f)

X_raw = data["data"]
y_raw = np.array(data["labels"])

max_len = max(len(np.array(seq).flatten()) for seq in X_raw)
X = np.array([
    np.pad(np.array(seq).flatten(), (0, max_len - len(np.array(seq).flatten())))
    for seq in X_raw
])
X = X[..., np.newaxis]  # add channel dimension for CNN

# --- Encode labels ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)
class_names = label_encoder.classes_

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- Define CNN model ---
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train model ---
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- Save model ---
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.h5")

# --- Predictions ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("CNN Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
save_plot(fig, "cnn_confusion_matrix.png")

# --- Classification Report ---
print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels, zero_division=0))

# --- Learning Curves ---
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
save_plot(fig, "cnn_learning_curves.png")

# --- t-SNE Plot ---
try:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_test.reshape(X_test.shape[0], -1))
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true_labels, palette='tab10', ax=ax)
    ax.set_title("t-SNE of Test Set")
    ax.legend(loc='best', fontsize='small')
    save_plot(fig, "cnn_tsne.png")
except Exception as e:
    print("t-SNE Error:", e)

# --- Error Analysis ---
errors = y_true != y_pred
error_counts = Counter(label_encoder.inverse_transform(y_true[errors]))
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
ax.bar(error_counts.keys(), error_counts.values())
ax.set_title("CNN Misclassifications per Class")
ax.set_xlabel("True Class")
ax.set_ylabel("Error Count")
save_plot(fig, "cnn_error_analysis.png")

# --- Grad-CAM Implementation ---
def grad_cam(model, sample):
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(sample)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-8)
    return heatmap, pred_index.numpy()

# Select a sample index to visualize Grad-CAM
sample_idx = 0
sample_input = np.expand_dims(X_test[sample_idx], axis=0)
heatmap, pred_idx = grad_cam(model, sample_input)

fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
ax.plot(sample_input[0], label='Input Signal')
ax.imshow(np.expand_dims(heatmap, axis=0), aspect='auto', cmap='jet', alpha=0.5,
          extent=(0, sample_input.shape[1], ax.get_ylim()[0], ax.get_ylim()[1]))
ax.set_title(f"Grad-CAM Heatmap for class: {label_encoder.inverse_transform([pred_idx])[0]}")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Signal Amplitude")
save_plot(fig, "cnn_gradcam.png")

print(f"\n✅ All CNN visualizations saved in: {cnn_vis_dir}")
