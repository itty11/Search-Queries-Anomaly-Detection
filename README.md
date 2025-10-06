# Search-Queries-Anomaly-Detection
This project detects **anomalous search query behavior** (e.g., unusual spikes or drops in clicks, impressions, or CTR) using an **Isolation Forest** machine learning model.   It includes both a **training script** and an **interactive Streamlit dashboard** for visualization and analysis.

### **Data Description**
Dataset Example (from [Aman Kharwal - The Clever Programmer]([https://thecleverprogrammer.com/](https://amanxai.com/2023/11/20/search-queries-anomaly-detection-using-python/))):

| Column | Description |
|---------|--------------|
| **Top queries** | Search terms queried on the site |
| **Clicks** | Number of clicks for the query |
| **Impressions** | How many times it appeared in search results |
| **CTR** | Click-through rate (percentage) |
| **Position** | Average search ranking position |

**Shape:** `(1000, 5)`  
**Detected anomalies:** 20 out of 1000 records (2.00%)

### **Key Features**
- Automatically **cleans CTR values** (removes “%” and converts to numeric).  
- Uses **Isolation Forest** for anomaly detection.  
- Scales numeric data with **StandardScaler**.  
- Visualizes anomalies using **PCA scatter plots**.  
- Displays feature correlations via a **heatmap**.  
- Provides a **Streamlit dashboard** for interactive detection.  
- Allows **CSV download** of detected anomalies.

  ### **Install dependencies**

  pip install pandas numpy scikit-learn seaborn matplotlib streamlit


### **Training Script**

Run Training (Detect Anomalies & Save Results)

python search_queries_anomaly_detection.py


### **Streamlit Dashboard**

Run the Dashboard

streamlit run app.py


# Dashboard Features
1. Upload & Preview

Upload your CSV dataset — the app shows the first few rows and detects anomalies automatically.

2. PCA Visualization

Interactive 2D scatter plot highlighting normal vs anomalous points:

 Blue → Normal queries

 Red → Detected anomalies

3. Feature Heatmap

Correlation heatmap among metrics like Clicks, Impressions, CTR, and Position.

4. Anomalies Table & Download

Displays top anomalous queries with option to:

Inspect directly in the app.

Download as a .csv file for further investigation.

# Tech Stack

Python 3.11+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit

Model: Isolation Forest (unsupervised anomaly detection)

Visualization: PCA, Correlation Heatmap

# Future Enhancements

Add time-series trend visualization (CTR or Clicks over time).

Integrate real-time anomaly alerts.

Deploy as a web service with scheduled updates.

Author

Ittyavira C Abraham

MCA AI Student @ Amrita Vishwa Vidyapeetham

Passionate about AI, ML, and intelligent automation systems.
