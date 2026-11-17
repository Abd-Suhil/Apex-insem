# CustomerIQ – Intelligent Customer Segmentation & Business Analytics

> **Turn Customer Data into Business Intelligence.**

CustomerIQ is an enterprise-grade Unsupervised Machine Learning and Business Intelligence platform built for customer segmentation, cohort profiling, real-time prospect scoring, and strategic revenue optimization.

---

## 🏛️ System Architecture

```
Apex minor project/
├── data/
│   ├── mall_customers.csv              # Canonical benchmark dataset (200 customer profiles)
│   ├── sample_customers.csv            # Cohort test dataset for batch prediction
│   └── sample_predictions.csv          # Pre-generated batch scoring output
├── models/
│   ├── kmeans_model.joblib             # Trained K-Means model (k-means++ initialization, K=5)
│   ├── preprocessor.joblib             # Fitted StandardScaler and PCA transformers
│   ├── cluster_metadata.json           # Centroids, unsupervised metrics, and business personas
│   └── training_summary.json           # Ingestion statistics and population distributions
├── src/
│   ├── __init__.py                     # Package declaration
│   ├── data_loader.py                  # Flexible schema normalization, validation, and stats
│   ├── preprocessor.py                 # Feature selection, StandardScaler, and PCA projection
│   ├── segmentation_model.py           # K-Means++ clustering, Elbow (WCSS) & Silhouette search
│   ├── evaluation.py                   # Silhouette Score, Davies-Bouldin, and Calinski-Harabasz metrics
│   ├── profiling.py                    # Dynamic persona mapping and commercial strategy generation
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── pdf_report.py               # Professional ReportLab executive PDF dossier generator
│   └── ui/
│       ├── __init__.py
│       ├── theme.py                    # Dark/Light SaaS theme styling and Plotly templates
│       ├── components.py               # KPI cards, section titles, banners, and empty states
│       └── pages/
│           ├── __init__.py
│           ├── overview.py             # Section 1: Executive Overview Dashboard
│           ├── data_explorer.py        # Section 2: Data Explorer & Filter Studio
│           ├── analytics.py            # Section 3: Exploratory Analytics (Plotly)
│           ├── clustering_studio.py    # Section 4: K-Means Clustering Studio
│           ├── segments.py             # Section 5: Customer Segments & Profiles
│           ├── prediction.py           # Section 6: Real-Time & Batch Prediction
│           ├── bi_insights.py          # Section 7: Commercial Business Intelligence
│           ├── model_performance.py    # Section 8: Unsupervised Model Diagnostics
│           ├── reports.py              # Section 9: Reports & PDF Dossier Exports
│           └── settings_page.py        # Section 10: Settings & Workspace Preferences
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                # Comprehensive unit and integration test suite
├── app.py                              # Streamlit multipage application entry point
├── train.py                            # Pipeline training and artifact export script
├── predict.py                          # CLI and programmatic inference engine
├── requirements.txt                    # Pinned project dependencies
└── README.md                           # System documentation and operational guide
```

---

## 🎨 Design System & UI/UX

Inspired by modern SaaS platforms such as Stripe, Linear, and Vercel:
* **Color Palette**: Modern dark navy/charcoal (`#0B0F19`, `#111827`, `#1F2937`) with purple, indigo, and blue accent highlights (`#6366F1`, `#8B5CF6`, `#3B82F6`).
* **Light Theme**: Fully supported with dynamic CSS injection via Settings.
* **Typography**: Clean, professional Plus Jakarta Sans hierarchy with subtle border lines, rounded cards, and responsive metric containers.
* **Interactive Visualizations**: High-performance Plotly charts with custom dark/light theme templates, hover tooltips, and interactive zoom/pan capabilities.
* **Navigation**: Sleek, distraction-free sidebar with 10 dedicated functional views.

---

## 🧭 Navigation & Modules

| Module | Purpose | Key Features |
| :--- | :--- | :--- |
| **1. Overview** | Executive Landing | KPI cards, income/spending distribution, portfolio donut chart, quick action routing. |
| **2. Data Explorer** | Dataset Management | CSV upload, schema verification, multi-parameter sliders, text search, duplicate/null audits. |
| **3. Exploratory Analytics** | Statistical Discovery | Univariate histograms, bivariate scatter plots, Pearson correlation heatmap, pairwise interaction matrix. |
| **4. Clustering Studio** | Machine Learning Studio | Feature selection, scaler choice, interactive Elbow Curve (WCSS) & Silhouette plots, on-the-fly retraining. |
| **5. Customer Segments** | Segment Deep Dive | Spatial scatter plot with gold-bordered star ($\bigstar$) centroids, demographic comparisons, persona cards. |
| **6. Customer Prediction** | Inference Engine | Single customer prospect sliders with spatial positioning, plus batch CSV cohort scoring and export. |
| **7. Business Intelligence** | Revenue Opportunities | Quadrant bubble matrix, purchasing power indices, data-backed commercial playbooks. |
| **8. Model Performance** | Algorithmic Diagnostics | Silhouette Score (0.6507), Davies-Bouldin Index (0.4569), Calinski-Harabasz Score (559.93), limitations audit. |
| **9. Reports & Exports** | Artifact Generation | High-resolution publication PNG charts, structured CSV exports, and **ReportLab executive PDF generation**. |
| **10. Settings** | System Preferences | Dark/Light theme toggle, active pipeline audit, cache reset, environment metadata. |

---

## 👥 Customer Personas Discovered

| Cluster | Segment Persona | Income Level | Spending Level | Commercial Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Youth Trendsetters / Enthusiasts** | Low | High | Social campaigns, influencer partnerships, flash discounts, BNPL financing. |
| **1** | **Mainstream / Everyday Balancers** | Moderate | Moderate | Seasonal promotions, loyalty tier progression, family bundle discounts. |
| **2** | **Budget Conscious / Thrifty** | Low | Low | Value-focused clearance events, bulk buy discounts, essential reward points. |
| **3** | **VIP / High Rollers** | High | High | Exclusive luxury previews, VIP concierge, bespoke loyalty rewards, high LTV focus. |
| **4** | **Careful Savers / High Potential** | High | Low | Emphasize durability, investment return, premium cashback, and product quality warranties. |

---

## 🚀 Quick Start Guide

### 1. Launch Streamlit Web Application
```bash
.\venv\Scripts\streamlit.exe run app.py
```
Access the application at: **`http://localhost:8501`**

### 2. Train the Model via CLI
```bash
python train.py --k 5 --features annual_income spending_score
```

### 3. Run Predictions
**Single Customer Scoring:**
```bash
python predict.py --income 85 --spending 80 --age 32
```

**Batch Cohort Scoring:**
```bash
python predict.py --batch-input data/sample_customers.csv --batch-output data/predictions.csv
```

### 4. Execute Test Suite
```bash
python tests/test_pipeline.py
```
Runs 7 unit and integration tests across data loading, preprocessor transformations, KMeans clustering, silhouette optimization, and prediction.
