import pickle
import numpy as np
from collections import Counter

# Load the dataset
with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = data_dict["data"]
labels = data_dict["labels"]

# Check data lengths
lengths = [len(sample) for sample in data]
unique_lengths = set(lengths)
print(f"🔎 Unique sample lengths: {unique_lengths}")

# Expected length: Most common length
expected_length = Counter(lengths).most_common(1)[0][0]
print(f"✅ Expected feature length: {expected_length}")

# Clean data: keep only samples with the expected length
cleaned_data = []
cleaned_labels = []
removed_indices = []

for i, sample in enumerate(data):
    if len(sample) == expected_length:
        cleaned_data.append(sample)
        cleaned_labels.append(labels[i])
    else:
        removed_indices.append(i)

print(f"🧹 Removed {len(removed_indices)} inconsistent samples")
print(f"📊 Cleaned dataset size: {len(cleaned_data)} samples")

# Save cleaned dataset
with open("data_cleaned.pickle", "wb") as f:
    pickle.dump({"data": cleaned_data, "labels": cleaned_labels}, f)

print("✅ Saved cleaned data to 'data_cleaned.pickle'")
