import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Setup directories for saving visualizations and models
VIS_DIR = "visualizations/transformer"
MODEL_DIR = "models"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def save_plot(fig, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(VIS_DIR, name), bbox_inches="tight")
    plt.close(fig)

# Load data
with open("data.pickle", "rb") as f:
    data = pickle.load(f)

print(f"Data keys: {list(data.keys())}")

# Extract raw data and labels
X_raw = data['data']   # likely list of arrays (ragged)
y_raw = np.array(data['labels'])

print(f"Number of samples: {len(X_raw)}")

# Pad sequences so that all have same length (pad on sequence length dimension)
# Assuming X_raw is list of (seq_len_i, feature_dim) arrays or lists
# First detect feature dimension (number of features per time step)
feature_dim = None
for sample in X_raw:
    arr = np.array(sample)
    if arr.ndim == 2:
        feature_dim = arr.shape[1]
        break
    elif arr.ndim == 1:
        feature_dim = 1
        break
if feature_dim is None:
    raise ValueError("Cannot infer feature dimension from data")

print(f"Inferred feature dimension: {feature_dim}")

# Convert all samples to arrays and pad sequences to max length
X_arr = []
for sample in X_raw:
    arr = np.array(sample)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)  # reshape 1D features to 2D with features=1
    X_arr.append(arr)

# Find max seq_len
max_seq_len = max(arr.shape[0] for arr in X_arr)
print(f"Maximum sequence length: {max_seq_len}")

# Pad sequences with zeros on the time dimension to max_seq_len
X_padded = pad_sequences(
    [arr for arr in X_arr],
    maxlen=max_seq_len,
    dtype='float32',
    padding='post',
    value=0.0
)  # Result: (samples, max_seq_len, feature_dim)

print(f"Padded data shape: {X_padded.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)
class_names = label_encoder.classes_
print(f"Classes found: {class_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout=0.1):
    feature_dim = inputs.shape[-1]
    key_dim = max(feature_dim // num_heads, 1)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    x_ff = layers.Dense(ff_dim, activation='relu')(x)
    x_ff = layers.Dense(feature_dim)(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + x_ff)
    return x

input_shape = X_train.shape[1:]  # (seq_len, feature_dim)
inputs = layers.Input(shape=input_shape)

x = transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout=0.1)
x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout=0.1)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

model.save(os.path.join(MODEL_DIR, "transformer_model.h5"))

# Predict and evaluate
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# Confusion Matrix
fig, ax = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("Transformer - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
save_plot(fig, "confusion_matrix.png")

# Classification report
print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels, zero_division=0))

# Learning curves
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Validation')
ax[0].set_title('Accuracy')
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Validation')
ax[1].set_title('Loss')
ax[1].legend()
save_plot(fig, "learning_curves.png")

# t-SNE visualization of embeddings
try:
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)
    embeddings = intermediate_layer_model.predict(X_test)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_true_labels, palette="tab10", ax=ax)
    ax.set_title("t-SNE Plot of Transformer Embeddings")
    save_plot(fig, "tsne.png")
except Exception as e:
    print("t-SNE error:", e)

# Error analysis
errors = y_true != y_pred
error_counts = Counter(label_encoder.inverse_transform(y_true[errors]))
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(error_counts.keys(), error_counts.values())
ax.set_title("Misclassifications per Class")
ax.set_xlabel("True Class")
ax.set_ylabel("Count")
save_plot(fig, "error_analysis.png")

print(f"\n✅ Transformer model training and visualization complete! Visualizations saved at '{VIS_DIR}'.")
