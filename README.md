# Retail & Transportation Machine Learning Capstone

**Course:** 23CSE301 Machine Learning — Capstone Project  
**Academic Year:** 2026–2027  
**Scope:** Review 1 (Full Regression Track + Classification Part A) & Review 2 (Classification Part B + Full Clustering Track + Bonus Web Application)

---

## 1. Problem Statements & Overview

This capstone project implements an end-to-end Machine Learning pipeline across three distinct analytical problem tracks:

1. **Regression Track (Bike Sharing Demand Forecasting):**
   - **Problem:** Forecast hourly rental demand (`cnt`) from temporal, seasonal, and environmental indicators (temperature, humidity, windspeed, weather situation).
   - **Algorithms Evaluated:** 10 regression algorithms (Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN).
   - **Best Model:** Tuned Random Forest Regressor ($R^2 = 0.947$, $\text{RMSE} = 41.52$, $\text{MAE} = 25.13$).

2. **Classification Track (Retail Customer Repeat Purchase Prediction):**
   - **Problem:** Predict whether an e-commerce customer will return as a **Repeat Buyer** ($\text{Frequency} > 1$) from their historical **Recency** (days elapsed) and **Monetary Value** (£ total net spend).
   - **Leakage Prevention:** Purchase frequency defines the ground-truth target and is strictly excluded from feature inputs $X$. An explicit `IQRClipper` caps predictor outliers on the training split only, followed by `StandardScaler`.
   - **Algorithms Evaluated:** All 10 classification algorithms across Part A (Review 1) and Part B (Review 2).
   - **Best Model:** Tuned Random Forest Classifier / Bagging Classifier ($\text{Accuracy} = 84.8\%$, $\text{ROC-AUC} = 0.905$).

3. **Clustering Track (Unsupervised Customer Segmentation):**
   - **Problem:** Discover organic behavioral customer segments from unlabelled RFM profiles (Recency, Frequency, Monetary Value) without using target labels during training.
   - **Algorithms Evaluated:** K-Means Clustering ($k=2 \dots 8$) and Agglomerative Hierarchical Clustering across 3 linkage strategies (`ward`, `complete`, `average`).
   - **Optimal Architecture:** K-Means ($k=4$, Silhouette Score = 0.3314, Davies-Bouldin = 1.0115, Calinski-Harabasz = 3,234.44).
   - **Personas Discovered:** *Champions & VIPs*, *At-Risk / Dormant Churners*, *Potential Loyalists*, and *Recent First-Timers*.

---

## 2. Consolidated Results Summary Tables

### 2.1 Classification Track — Consolidated 10-Algorithm Comparison (Review 2 Deliverable D6 & Rubric A2)

All 10 algorithms trained on the identical 80/20 stratified split (`X_train, y_train`) and evaluated on held-out test data:

| Rank | Algorithm | Track | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Bagging Classifier** | **Part B** | **0.8482** | **0.8461** | **0.8482** | **0.8465** | **0.9048** |
| **2** | **Random Forest Classifier** | **Part B** | **0.8447** | **0.8430** | **0.8447** | **0.8435** | **0.9044** |
| **3** | **MLP Classifier (Neural Net)** | **Part B** | **0.8447** | **0.8426** | **0.8447** | **0.8430** | **0.9065** |
| 4 | Decision Tree Classifier | Part A | 0.8401 | 0.8375 | 0.8401 | 0.8375 | 0.8944 |
| 5 | AdaBoost Classifier | Part B | 0.8389 | 0.8380 | 0.8389 | 0.8384 | 0.8996 |
| 6 | Support Vector Machine (SVC) | Part A | 0.8366 | 0.8354 | 0.8366 | 0.8359 | 0.8790 |
| 7 | Gradient Boosting Classifier | Part B | 0.8355 | 0.8328 | 0.8355 | 0.8331 | 0.8994 |
| 8 | Logistic Regression | Part A | 0.8331 | 0.8316 | 0.8331 | 0.8322 | 0.8917 |
| 9 | K-Nearest Neighbors | Part A | 0.8308 | 0.8284 | 0.8308 | 0.8290 | 0.8700 |
| 10 | Naive Bayes (Gaussian) | Part A | 0.8216 | 0.8405 | 0.8216 | 0.8256 | 0.8655 |

#### Classification Hyperparameter Tuning (Rubric A3)
- **Model Selected:** Random Forest Classifier
- **GridSearchCV Optimal Parameters:** `{'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 2, 'n_estimators': 50}`
- **Metric Improvement:** ROC-AUC increased from `0.9044` to `0.9053`; tree complexity regularized from unconstrained depth to `max_depth=4`, mitigating overfitting.

---

### 2.2 Clustering Track — Consolidated Metrics Table (Review 2 Rubric B2)

| Algorithm | Clusters ($k$) | Linkage | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means (Optimal)** | **4** | **Centroid** | **0.3314** | **1.0115** | **3,234.44** |
| Agglomerative Clustering | 4 | Ward | 0.2778 | 1.1008 | 2,826.89 |
| Agglomerative Clustering | 4 | Average | 0.3231 | 0.6935 | 603.63 |
| Agglomerative Clustering | 4 | Complete | 0.1134 | 1.5222 | 901.58 |

#### Strategic Customer Personas Discovered (Post-Hoc Ground-Truth Validation)
| Cluster | Persona Name | Customer Count | Median Recency | Median Frequency | Median Monetary | Repeat Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Potential Loyalists** | 945 | 21.0 days | 2 invoices | £457.7 | 62.0% |
| **1** | **Champions & VIPs** | 778 | 9.5 days | 9 invoices | £3,416.3 | **100.0%** |
| **2** | **At-Risk / Dormant Churners** | 1,378 | 187.0 days | 1 invoice | £258.6 | 24.9% |
| **3** | **Steady Regular Customers** | 1,211 | 64.0 days | 4 invoices | £1,362.0 | **97.9%** |

---

### 2.3 Regression Track — 10 Algorithms Summary (Review 1 Track)

| Rank | Model | $R^2$ Score | RMSE | MAE | 5-Fold CV $R^2$ (Mean ± Std) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Regressor (Tuned)** | **0.9472** | **41.52** | **25.13** | **0.9431 ± 0.0038** |
| 2 | Gradient Boosting Regressor | 0.9238 | 49.91 | 33.74 | 0.9205 ± 0.0042 |
| 3 | Decision Tree Regressor | 0.8874 | 60.77 | 39.86 | — |
| 4 | K-Nearest Neighbors Regressor | 0.8412 | 72.18 | 48.91 | — |
| 5 | Polynomial Regression (Degree 2) | 0.6841 | 101.42 | 74.20 | — |
| 6 | Support Vector Regressor (SVR) | 0.6528 | 106.31 | 75.14 | — |
| 7 | Ridge Regression | 0.6289 | 109.91 | 81.33 | — |
| 8 | Linear Regression | 0.6288 | 109.92 | 81.34 | — |
| 9 | ElasticNet Regression | 0.5841 | 116.35 | 87.21 | — |
| 10 | Lasso Regression | 0.5812 | 116.76 | 87.54 | — |

---

## 3. Repository Architecture

```text
Retail_ML_Capstone/
├── README.md                               # Master documentation, architecture, results tables
├── requirements.txt                        # Pinned dependencies for full replication
├── data/
│   ├── online_retail_II.xlsx               # Raw retail transactional dataset (Classification & Clustering)
│   └── hour.csv                            # Bike sharing hourly dataset (Regression Track)
├── notebooks/
│   ├── 1_EDA.ipynb                         # Master Retail ML Capstone (EDA, Classification Part A & B, Full Clustering Track)
│   └── 2_regression_track.ipynb            # Bike Sharing Regression Track (10 Models, Tuning, Residuals)
├── models/
│   ├── best_classification_pipeline.joblib # Serialized production Random Forest pipeline
│   └── best_clustering_pipeline.joblib     # Serialized K-Means scaler + model + PCA pipeline
├── outputs/                                # High-resolution evaluation charts, ROC curves, CSV matrices
│   ├── classification_consolidated_10_models.csv
│   ├── classification_part_b_results.csv
│   ├── classification_tuning_improvement.csv
│   ├── classification_roc_curves_combined.png
│   ├── classification_part_b_confusion_matrices.png
│   ├── classification_ensemble_feature_importances.png
│   ├── clustering_metrics_comparison.csv
│   ├── clustering_kmeans_elbow_curve.png
│   ├── clustering_dendrogram.png
│   ├── clustering_pca_2d.png
│   ├── clustering_tsne_2d.png
│   ├── clustering_persona_profiles.csv
│   └── clustering_rfm_personas.png
└── app/                                    # Review 2 Bonus Track (+2 Marks): Web Application
    ├── app.py                              # Streamlit GUI for inference and interactive customer segmentation
    ├── requirements.txt                    # Standalone deployment dependencies
    └── README.md                           # Cloud deployment guide (Streamlit Cloud / HF Spaces)
```

---

## 4. Setup & Execution Instructions

### 4.1 Environment Setup
1. Clone or open the repository:
   ```bash
   git clone <repo-url>
   cd Retail_ML_Capstone
   ```
2. Create and activate a Python virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### 4.2 Running Jupyter Notebooks
Start Jupyter Lab / Notebook:
```bash
jupyter lab
```
Run the notebooks top-to-bottom:
- `notebooks/1_EDA.ipynb`: Master Retail ML Capstone (EDA, Classification Track Part A & Part B, and Full Clustering Track)
- `notebooks/2_regression_track.ipynb`: Bike Sharing Regression Track (10 Models, Tuning, Residuals)

### 4.3 Launching the Interactive GUI (Bonus Track)
To run the Streamlit web application locally:
```bash
streamlit run app/app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 5. Academic Integrity & Tools Citation
- **Primary Frameworks:** `Python 3`, `scikit-learn`, `pandas`, `NumPy`, `SciPy`, `Matplotlib`, `Seaborn`, `Streamlit`.
- Preprocessing pipelines (`IQRClipper`, `StandardScaler`), cross-validation splits, and clustering metrics adhere strictly to the **23CSE301 Capstone Project Guidelines**.
