import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import warnings

# Suppress matplotlib UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------
# Load processed data
# ---------------------
with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)

raw_data = dataset['data']
labels = np.array(dataset['labels'])

# ---------------------
# Filter samples with consistent length
# ---------------------
lengths = [len(sample) for sample in raw_data]
length_counts = Counter(lengths)
correct_length = length_counts.most_common(1)[0][0]
print(f"Using samples of length: {correct_length}")

filtered_data = [sample for sample in raw_data if len(sample) == correct_length]
filtered_labels = [label for sample, label in zip(raw_data, labels) if len(sample) == correct_length]

data = np.array(filtered_data)
labels = np.array(filtered_labels)

print(f"Filtered data shape: {data.shape}")
print(f"Number of landmarks per sample: {data.shape[1]//2}")

# ---------------------
# 1. Scatter plot (first sample)
# ---------------------
sample = data[0]
x = sample[::2]
y = sample[1::2]

plt.figure(figsize=(6,6))
plt.scatter(x, y, color='blue')
plt.gca().invert_yaxis()
plt.title(f"Scatter Plot of Landmarks - Sample 0 (Label: {labels[0]})")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_sample0.png", dpi=300)
plt.show()

# ---------------------
# 2. Overlay scatter plots of ALL classes
# ---------------------
unique_labels = list(set(labels))
plt.figure(figsize=(10,10))

colors = sns.color_palette("hsv", len(unique_labels))
for i, label in enumerate(unique_labels):
    idx = np.where(labels == label)[0][0]
    sample = data[idx]
    x = sample[::2]
    y = sample[1::2]
    plt.scatter(x, y, label=f"{label}", color=colors[i], alpha=0.7)

plt.gca().invert_yaxis()
plt.title("Overlay Scatter Plot: One Sample Per Class (All Classes)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("scatter_overlay_all.png", dpi=300)
plt.show()

# ---------------------
# 3. Histogram of all X and Y
# ---------------------
all_x = data[:, ::2].flatten()
all_y = data[:, 1::2].flatten()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(all_x, bins=30, color='blue', alpha=0.7)
plt.title("Histogram of all X coordinates")
plt.xlabel("X value")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(all_y, bins=30, color='green', alpha=0.7)
plt.title("Histogram of all Y coordinates")
plt.xlabel("Y value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("histogram_xy.png", dpi=300)
plt.show()

# ---------------------
# 4. Boxplot of X and Y coordinates by landmark index
# ---------------------
rows = []
num_landmarks = data.shape[1] // 2
for sample in data:
    for i in range(num_landmarks):
        rows.append({"landmark": i, "coord": "X", "value": sample[2*i]})
        rows.append({"landmark": i, "coord": "Y", "value": sample[2*i+1]})

df = pd.DataFrame(rows)

plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='landmark', y='value', hue='coord')
plt.title("Boxplot of X and Y coordinates by Landmark Index")
plt.xlabel("Landmark Index")
plt.ylabel("Normalized Coordinate Value")
plt.tight_layout()
plt.savefig("boxplot_landmarks.png", dpi=300)
plt.show()

# ---------------------
# 5. PCA Plot
# ---------------------
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

plt.figure(figsize=(8,6))
sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1], hue=labels, palette="tab10", legend='full')
plt.title("PCA of Hand Landmark Features")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("pca_plot.png", dpi=300)
plt.show()

# ---------------------
# 6. t-SNE Plot
# ---------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data)

plt.figure(figsize=(8,6))
sns.scatterplot(x=data_tsne[:,0], y=data_tsne[:,1], hue=labels, palette="tab10", legend='full')
plt.title("t-SNE of Hand Landmark Features")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("tsne_plot.png", dpi=300)
plt.show()

# ---------------------
# 7. Correlation Heatmap
# ---------------------
df_features = pd.DataFrame(data)
corr = df_features.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Heatmap of Landmark Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# ---------------------
# 8. Class-wise Stats: Mean & Std
# ---------------------
df_data = pd.DataFrame(data)
df_data['label'] = labels

mean_per_class = df_data.groupby('label').mean()
std_per_class = df_data.groupby('label').std()

print("\n✅ Mean feature values per class:")
print(mean_per_class.head())

print("\n✅ Standard deviation of features per class:")
print(std_per_class.head())

# Save stats
mean_per_class.to_csv("mean_features_per_class.csv")
std_per_class.to_csv("std_features_per_class.csv")
