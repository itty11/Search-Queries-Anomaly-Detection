import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Load Dataset 
df = pd.read_csv("Queries.csv")  
print("\n=== Dataset Loaded ===")
print(f"Shape: {df.shape}")
print(df.head())

# Data Cleaning 
df = df.dropna()
print(f"\nAfter dropping NaN → {df.shape}")

# Convert CTR to numeric
if "CTR" in df.columns:
    df["CTR"] = df["CTR"].astype(str).str.replace("%", "").astype(float)

# Select numeric features
features = ["Clicks", "Impressions", "CTR", "Position"]
df_features = df[features]

# Scale Data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Train Isolation Forest 
model = IsolationForest(contamination=0.02, random_state=42)
df["anomaly_score"] = model.fit_predict(X_scaled)
df["anomaly_score"] = df["anomaly_score"].map({1: 0, -1: 1})

num_anomalies = df["anomaly_score"].sum()
print(f"\n Anomalies detected: {num_anomalies}/{len(df)} ({num_anomalies/len(df)*100:.2f}%)")

# PCA Visualization 
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = pca_result[:, 0], pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="pca1", y="pca2",
    hue="anomaly_score", palette={0: "blue", 1: "red"},
    data=df, alpha=0.7
)
plt.title("Search Query Anomalies (PCA Visualization)")
plt.show()

# Feature Correlation 
plt.figure(figsize=(6, 4))
sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Save Detected Anomalies 
anomalies = df[df["anomaly_score"] == 1]
anomalies.to_csv("detected_anomalies.csv", index=False)
print("\n Saved detected anomalies → detected_anomalies.csv")
