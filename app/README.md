# Retail Customer Intelligence Platform — Interactive GUI

This web application provides a production-ready interactive interface for the **23CSE301 Machine Learning Capstone** (Review 2 Bonus Track: Deliverable D8).

## Features
- **Repeat Customer Predictor (Classification Track):** Accepts customer Recency and Monetary spend inputs, executes real-time inference via our trained Random Forest pipeline (`models/best_classification_pipeline.joblib`), outputs repeat-buyer probability, confidence bar, and strategic marketing actions.
- **Customer Segmentation (Clustering Track):** Accepts Recency, Frequency, and Monetary parameters, maps customer into optimal K-Means behavioral segments ($k=4$), renders an interactive 2D PCA projection with customer marker ⭐, and displays persona playbooks.
- **Capstone Leaderboard:** Live comparative performance tables for all 10 classification algorithms, clustering linkage evaluations, and regression track models.

## How to Run Locally
From the project root directory (`Retail_ML_Capstone/`), execute:
```bash
streamlit run app/app.py
```

## How to Deploy to Public Cloud (Bonus +1 Mark)
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set Main file path to `app/app.py`.
5. Click **Deploy**!
