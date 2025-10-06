import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

st.set_page_config(page_title="Search Query Anomaly Detection", layout="wide")

st.title(" Search Queries Anomaly Detection Dashboard")
st.markdown("Detect unusual search query behavior using **Isolation Forest** and PCA visualization.")

uploaded_file = st.file_uploader(" Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    #  Load Data 
    df = pd.read_csv(uploaded_file)
    st.success(f" Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())

    # Data Cleaning 
    df = df.dropna()

    # Clean CTR 
    if "CTR" in df.columns:
        df["CTR"] = df["CTR"].astype(str).str.replace("%", "").astype(float)

    # Features for anomaly detection
    features = ["Clicks", "Impressions", "CTR", "Position"]
    df_features = df[features]

    #  Scale Data 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    #  Isolation Forest Model 
    model = IsolationForest(contamination=0.02, random_state=42)
    df["Anomaly"] = model.fit_predict(X_scaled)
    df["Anomaly"] = df["Anomaly"].map({1: 0, -1: 1})

    num_anomalies = df["Anomaly"].sum()
    st.write(f" **Detected {num_anomalies} anomalies out of {len(df)} records**")

    #  PCA Visualization 
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df["pca1"], df["pca2"] = pca_result[:, 0], pca_result[:, 1]

    st.subheader(" PCA Visualization (Anomalies Highlighted)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="Anomaly",
        palette={0: "blue", 1: "red"},
        data=df, alpha=0.7, ax=ax
    )
    st.pyplot(fig)

    # Feature Correlation Heatmap 
    st.subheader(" Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # Display and Download Anomalies 
    st.subheader(" Detected Anomalous Queries")
    anomalies = df[df["Anomaly"] == 1]
    st.dataframe(anomalies.head(10))

    csv = anomalies.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Anomalies CSV", csv, "detected_anomalies.csv", "text/csv")

else:
    st.info(" Please upload a dataset (CSV) to begin anomaly detection.")
