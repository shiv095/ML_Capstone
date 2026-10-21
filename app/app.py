import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Retail ML Capstone — Customer Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Path resolution
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Fallback paths if app is run directly from repo root
if not MODELS_DIR.exists():
    MODELS_DIR = Path("models")
if not OUTPUTS_DIR.exists():
    OUTPUTS_DIR = Path("outputs")

# Helper class for unpickling the IQRClipper
from sklearn.base import BaseEstimator, TransformerMixin
class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.clip(self.lower_, self.upper_, axis=1).to_numpy()

# Load models and artifacts with caching
@st.cache_resource
def load_classification_pipeline():
    model_path = MODELS_DIR / "best_classification_pipeline.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_clustering_pipeline():
    model_path = MODELS_DIR / "best_clustering_pipeline.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_table(csv_name):
    csv_path = OUTPUTS_DIR / csv_name
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# Sidebar branding & navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shopping-bag.png", width=110)
    st.markdown("## **Retail ML Capstone**")
    st.markdown("**Course:** 23CSE301 Machine Learning")
    st.markdown("**Review 2:** Part B Classification & Full Clustering")
    st.markdown("---")

    st.markdown("### **System Status**")
    clf_model = load_classification_pipeline()
    cl_model = load_clustering_pipeline()

    if clf_model is not None:
        st.success(" Classification Pipeline Loaded")
    else:
        st.warning("⚠️ Classification Pipeline Not Found")

    if cl_model is not None:
        st.success(" Clustering Pipeline Loaded")
    else:
        st.warning("⚠️ Clustering Pipeline Not Found")

    st.markdown("---")
    st.markdown("### **Quick Presets**")
    preset = st.selectbox(
        "Load Sample Customer Profile:",
        ["None (Manual Input)", "VIP Champion", "At-Risk Dormant", "Potential Loyalist", "New First-Timer"]
    )

# Header
st.markdown('<div class="main-header">🛍️ Retail Customer Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production ML Inference & Customer Segmentation Suite | Online Retail II Dataset</div>', unsafe_allow_html=True)

# Preset values
default_recency = 30
default_monetary = 750.0
default_frequency = 3

if preset == "VIP Champion":
    default_recency = 10
    default_monetary = 4250.0
    default_frequency = 14
elif preset == "At-Risk Dormant":
    default_recency = 240
    default_monetary = 180.0
    default_frequency = 1
elif preset == "Potential Loyalist":
    default_recency = 45
    default_monetary = 1100.0
    default_frequency = 4
elif preset == "New First-Timer":
    default_recency = 12
    default_monetary = 320.0
    default_frequency = 1

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Repeat Customer Predictor",
    "👥 Customer Segmentation (Clustering)",
    "📊 Capstone Model Leaderboard"
])

# -------------------------------------------------------------
# TAB 1: CLASSIFICATION
# -------------------------------------------------------------
with tab1:
    st.markdown("### 🎯 Supervised Classification — Repeat Customer Prediction")
    st.write("Predict whether a customer will return as a **Repeat Buyer ($Frequency > 1$)** based on their purchase **Recency** and total **Monetary Spend**.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### **Customer Behavioral Inputs**")
        recency_input = st.slider(
            "Recency (Days since last purchase):",
            min_value=1, max_value=375, value=int(default_recency), step=1,
            help="Number of days elapsed between customer's most recent purchase and benchmark date."
        )
        monetary_input = st.number_input(
            "Total Monetary Value (£):",
            min_value=1.0, max_value=50000.0, value=float(default_monetary), step=50.0,
            help="Cumulative gross transaction value generated by the customer in British Pounds."
        )

        predict_btn = st.button("🚀 Predict Repeat Customer Status", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### **Inference Output & Recommendation**")

        if clf_model is not None:
            # Prepare input dataframe
            input_df = pd.DataFrame([{
                'Recency': recency_input,
                'Monetary': monetary_input
            }])

            # Make prediction
            prob = clf_model.predict_proba(input_df)[0, 1]
            pred = int(prob >= 0.5)

            if pred == 1:
                st.success("### 🎉 Prediction: REPEAT CUSTOMER (Class 1)")
                st.metric(label="Repeat Purchase Probability", value=f"{prob * 100:.1f}%")
                st.progress(prob)

                st.markdown("""
                **💡 Strategic Recommendation:**
                - **Retention Strategy:** Customer exhibits high momentum. Enroll in loyalty tier rewards.
                - **Cross-Selling:** Push personalized bundle promotions and early-access catalogs.
                """)
            else:
                st.error("### ⚠️ Prediction: SINGLE-PURCHASE BUYER (Class 0)")
                st.metric(label="Repeat Purchase Probability", value=f"{prob * 100:.1f}%")
                st.progress(prob)

                st.markdown("""
                **💡 Strategic Recommendation:**
                - **Churn Risk:** Customer shows extended dormancy or insufficient basket depth.
                - **Win-Back Action:** Deploy targeted reactivation incentives (e.g. 15% discount on next purchase).
                """)
        else:
            st.info("Please train and export `best_classification_pipeline.joblib` using the capstone notebook.")

# -------------------------------------------------------------
# TAB 2: CLUSTERING
# -------------------------------------------------------------
with tab2:
    st.markdown("### 👥 Unsupervised Customer Segmentation (RFM Clustering)")
    st.write("Assign customers into data-driven behavioral personas using our optimal **K-Means ($k=4$)** model trained on normalized RFM profiles.")

    ccol1, ccol2 = st.columns([1, 1.2])

    with ccol1:
        st.markdown("#### **Customer RFM Inputs**")
        cl_recency = st.slider("Recency (Days):", 1, 375, int(default_recency), key="cl_r")
        cl_frequency = st.slider("Frequency (Distinct Invoices):", 1, 50, int(default_frequency), key="cl_f")
        cl_monetary = st.number_input("Monetary Value (£):", 1.0, 50000.0, float(default_monetary), step=50.0, key="cl_m")

        segment_btn = st.button("🔍 Assign Customer Segment", type="primary", use_container_width=True)

    with ccol2:
        st.markdown("#### **Segment Assignment & Persona Profile**")

        if cl_model is not None:
            scaler = cl_model['scaler']
            kmeans = cl_model['kmeans_model']
            pca = cl_model['pca_model']
            persona_df = cl_model.get('persona_summary', None)

            # Preprocess query point: log1p then scaler
            query_raw = np.array([[cl_recency, cl_frequency, cl_monetary]])
            query_log = np.log1p(query_raw)
            query_scaled = scaler.transform(query_log)

            # Predict cluster
            cluster_id = int(kmeans.predict(query_scaled)[0])

            # Project onto PCA 2D space
            query_pca = pca.transform(query_scaled)

            # Segment mappings based on RFM profile
            persona_names = {
                0: ("Potential Loyalists", "🌱 Recent buyers with moderate spend and growing purchase frequency."),
                1: ("Champions & VIPs", "🏆 Highest lifetime value, top purchase frequency, and recent activity. >95% repeat rate."),
                2: ("At-Risk / Dormant Churners", "⚠️ Long inactivity (>180 days) with low transaction count. High risk of permanent churn."),
                3: ("Steady Loyal Customers", "📈 Consistent repeat purchasing behavior with solid annual basket sizes.")
            }

            p_title, p_desc = persona_names.get(cluster_id, (f"Cluster {cluster_id}", "Custom customer behavioral segment."))

            st.info(f"### Assigned Segment: **{p_title}** (Cluster {cluster_id})")
            st.write(p_desc)

            # Plot PCA visualization
            st.markdown("##### **Customer Position in 2D PCA Space**")
            fig, ax = plt.subplots(figsize=(6.5, 4.2))

            # Dummy points representing clusters
            centers_pca = pca.transform(kmeans.cluster_centers_)
            palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            for c_idx in range(4):
                ax.scatter(centers_pca[c_idx, 0], centers_pca[c_idx, 1],
                           s=220, color=palette[c_idx], label=f'Cluster {c_idx}', alpha=0.6, edgecolors='black')

            # Highlight user query point
            ax.scatter(query_pca[0, 0], query_pca[0, 1],
                       s=350, color='crimson', marker='*', edgecolors='gold', linewidth=1.5,
                       label='Current Customer ⭐', zorder=10)

            ax.set_title("Customer Projection vs. Cluster Centroids (PCA Space)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend(loc='best', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Clustering pipeline `best_clustering_pipeline.joblib` not found. Please run the clustering track notebook.")

# -------------------------------------------------------------
# TAB 3: LEADERBOARD
# -------------------------------------------------------------
with tab3:
    st.markdown("### 📊 Capstone Model Leaderboards & Review 2 Deliverables")

    st.markdown("#### **1. Consolidated 10-Algorithm Classification Comparison (Review 2 Deliverable D6)**")
    df_cls = load_table("classification_consolidated_10_models.csv")
    if df_cls is not None:
        st.dataframe(
            df_cls.style.format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}',
                'Weighted F1': '{:.4f}', 'ROC-AUC': '{:.4f}'
            }).background_gradient(subset=['Accuracy', 'ROC-AUC'], cmap='Blues'),
            use_container_width=True
        )
    else:
        st.info("Table `classification_consolidated_10_models.csv` not found in outputs/.")

    st.markdown("---")
    st.markdown("#### **2. Consolidated Clustering Evaluation Table (Review 2 Rubric B2)**")
    df_clu = load_table("clustering_metrics_comparison.csv")
    if df_clu is not None:
        st.dataframe(
            df_clu.style.format({
                'Silhouette Score': '{:.4f}',
                'Davies-Bouldin Index': '{:.4f}',
                'Calinski-Harabasz Index': '{:.2f}'
            }).background_gradient(subset=['Silhouette Score', 'Calinski-Harabasz Index'], cmap='Blues'),
            use_container_width=True
        )
    else:
        st.info("Table `clustering_metrics_comparison.csv` not found in outputs/.")

    st.markdown("---")
    st.markdown("#### **3. Regression Track Summary (Review 1 Track)**")
    df_reg = load_table("regression_results_full.csv")
    if df_reg is not None:
        st.dataframe(df_reg, use_container_width=True)
    else:
        st.info("Table `regression_results_full.csv` not found in outputs/.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>23CSE301 Machine Learning Capstone | Review 2 Submission</div>", unsafe_allow_html=True)
