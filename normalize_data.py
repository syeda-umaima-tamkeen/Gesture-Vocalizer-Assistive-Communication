import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ===== Step 1: Load the cleaned data =====
data_path = "./data_cleaned.pickle"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ '{data_path}' not found. Please run the cleaning step first.")

with open(data_path, "rb") as f:
    data_dict = pickle.load(f)

raw_data = data_dict["data"]
labels = data_dict["labels"]

# ===== Step 2: Convert to NumPy array =====
data_array = np.array(raw_data)

# ===== Step 3: Normalize the features =====
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_array)

# ===== Step 4: Save normalized data =====
normalized_path = "data_normalized.pickle"
with open(normalized_path, "wb") as f:
    pickle.dump({"data": normalized_data, "labels": labels}, f)

print("✅ Normalization completed.")
print(f"🔁 Input shape: {data_array.shape}")
print(f"📦 Normalized data saved to: {normalized_path}")
print(f"🔍 Feature range after normalization: Min = {normalized_data.min():.4f}, Max = {normalized_data.max():.4f}")
