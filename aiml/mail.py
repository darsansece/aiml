import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load and clean dataset
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

# Fix gender case and convert to numeric
df["Gender"] = df["Gender"].str.lower().map({"female": 0, "male": 1})

# -----------------------------
# 2️⃣ Define features to use
# -----------------------------
features = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Standardize features
x = StandardScaler().fit_transform(df[features])

# -----------------------------
# 3️⃣ Apply KMeans clustering
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df["clusters"] = kmeans.fit_predict(x)

# -----------------------------
# 4️⃣ Visualize with t-SNE
# -----------------------------
tsne = TSNE(n_components=2, random_state=42)
x_embedded = tsne.fit_transform(x)

plt.figure(figsize=(8, 6))
plt.scatter(x_embedded[:, 0], x_embedded[:, 1],
            c=df["clusters"], cmap='tab10', s=60)
plt.title("Customer Segmentation with KMeans and t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
