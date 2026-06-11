# GenZ Social Media Addiction Predictor

## Overview

This project develops a machine learning solution to predict social media addiction levels among Generation Z users. By analysing behavioural and demographic patterns — including daily usage hours, platform preferences, and mental health indicators — the model classifies users as **Low**, **Medium**, or **High** addiction risk. The insights can guide digital wellness interventions, platform design decisions, and public health strategies targeting younger populations.

## Business Problem

Social media addiction in Gen Z is a growing concern linked to declining mental health, reduced academic performance, and disrupted sleep. Identifying at-risk users early enables targeted intervention before harm escalates. This project answers the question: **given a user's platform behaviour and demographics, what is their addiction risk level?**

## Dataset

| Field | Detail |
|---|---|
| Source | [Gen Z Social Media Usage Dataset — Kaggle](https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset) |
| Records | 1,000,000 |
| Features | 11 (9 input features after target removal) |
| Target Variable | `addiction_level` — Ordinal: Low / Medium / High |

**Feature breakdown:**

| Feature | Type | Description |
|---|---|---|
| `age` | int | User age (13–27) |
| `gender` | categorical | Male / Female / Other |
| `country` | categorical | 7 countries (India, USA, Canada, UK, Germany, Brazil, Australia) |
| `daily_usage_hours` | float | Average hours per day on social media |
| `primary_platform` | categorical | Instagram, YouTube, TikTok, Twitter, Snapchat |
| `num_platforms_used` | int | Number of platforms actively used |
| `purpose` | categorical | Entertainment, Socialising, Education, News, Content Creation |
| `avg_session_minutes` | float | Average session length in minutes |
| `night_usage` | binary | Whether user typically uses social media at night (0/1) |
| `mental_health_score` | float | Self-reported mental health score |
| `screen_time_before_sleep` | float | Minutes of screen time before sleeping |

**Target class distribution:**

| Class | Count | Share |
|---|---|---|
| Medium | 589,843 | 59.0% |
| Low | 251,220 | 25.1% |
| High | 158,937 | 15.9% |

> **Download the dataset:** [kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset](https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset)
> The dataset is not included in this repository due to its size (1M rows). Download it from Kaggle and place it in the project root before running the notebook.

## Technologies Used

| Category | Libraries |
|---|---|
| Data manipulation | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistics | SciPy |
| Machine learning | Scikit-Learn, XGBoost |
| Notebook | Jupyter Notebook |
| Model persistence | joblib |

## Project Workflow

### 1. Data Collection
The dataset (`genz_social_media_usage_1M.xls`) contains 1,000,000 rows of Gen Z social media behaviour. It was loaded with `pandas.read_csv` (despite the `.xls` extension, the file is comma-separated plain text).

### 2. Data Cleaning
- Verified zero duplicate rows and zero null values across all 12 columns
- Confirmed no type mismatches (no numeric values stored as objects)
- Checked for outliers using IQR method across all six numeric columns; extremes were **Winsorised** (capped) rather than removed to preserve the full 1M-row dataset

### 3. Exploratory Data Analysis
Key EDA findings:
- **Age** is nearly uniform across 13–27 (mean ≈ 20, low skew)
- **daily_usage_hours** and **avg_session_minutes** are the strongest numeric separators of addiction level
- High-addiction users show markedly higher `screen_time_before_sleep` and lower `mental_health_score`
- **Instagram** has the highest user share (~30%); **Entertainment** is the dominant usage purpose (~40%)
- Gender is near-equal split (Male ≈ Female ≈ 48%, Other ≈ 4%)
- Pearson correlations between numeric features are weak-to-moderate — no multicollinearity detected (no pair exceeded |r| > 0.85)

### 4. Feature Engineering
- **Target encoding**: `addiction_level` mapped ordinally → Low = 0, Medium = 1, High = 2
- **One-hot encoding**: `gender`, `country`, `primary_platform`, `purpose` — with `drop_first=True` to avoid the dummy variable trap
- **Train/test split**: 80/20, stratified by target to preserve class proportions
- **Scaling**: RobustScaler applied to the six numeric columns (fit on train only to prevent leakage) — chosen over StandardScaler due to residual tail distributions post-Winsorisation

### 5. Model Building

Three classifiers were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model; `max_iter=1000`, `n_jobs=-1` |
| Random Forest | 200 trees, used for both prediction and feature importance; `n_jobs=-1` |
| XGBoost | Final model; 200 estimators, `max_depth=6`, `learning_rate=0.1` |

Feature selection was performed using Random Forest importances — features with importance < 1% were dropped before training the final XGBoost model.

### 6. Model Evaluation
- **Classification report**: precision, recall, F1 per class (Low / Medium / High)
- **Confusion matrices**: side-by-side for Logistic Regression and Random Forest
- **ROC / AUC curves**: one curve per class (OvR), computed for XGBoost
- **Cross-validation**: 5-fold CV on XGBoost using macro F1 scoring

### 7. Insights

- `daily_usage_hours`, `avg_session_minutes`, and `screen_time_before_sleep` are the top predictors of addiction level
- `night_usage` adds signal beyond simple usage volume
- `mental_health_score` shows an inverse relationship with addiction level — higher addiction correlates with lower self-reported mental health
- Platform type and usage purpose contribute less than behavioural time-based features

## Visualisations

The notebook generates the following plots:

| Visualisation | Description |
|---|---|
| Distribution plots | Histogram + KDE for `age`, boxplot for `daily_usage_hours`, QQ plot for `avg_session_minutes` |
| Platform bar chart | Count of users per `primary_platform` |
| Addiction level pie chart | Class proportions of the target variable |
| Boxplots (6-panel) | Each numeric feature split by addiction level |
| KDE plots (6-panel) | Density distributions of numeric features per addiction class |
| Countplots | `gender`, `primary_platform`, `purpose` vs addiction level |
| Scatter plots | `daily_usage_hours` vs `avg_session_minutes` and vs `mental_health_score`, coloured by addiction level |
| Correlation heatmap | Pearson correlations among all numeric features |
| Pivot heatmaps | Avg daily usage by Purpose × Platform; % High addiction by Gender × Platform |
| Violin plots | `daily_usage_hours` and `mental_health_score` by addiction level |
| Outlier panels | Boxplots + histograms for all 6 numeric features |
| QQ plots | Normality check with skewness annotation for each numeric feature |
| Feature importance bar chart | Top 15 features ranked by Random Forest importance |
| Confusion matrices | Side-by-side for Logistic Regression and Random Forest |
| ROC curves | Per-class AUC for XGBoost |


## Future Improvements

- **Hyperparameter tuning**: Grid search or Optuna for XGBoost `max_depth`, `learning_rate`, `subsample`
- **Model deployment**: Flask / FastAPI REST endpoint or Streamlit dashboard for real-time prediction
- **Automated pipelines**: sklearn `Pipeline` object to bundle encoding + scaling + model for cleaner inference
- **Real-time predictions**: Integration with platform APIs or mobile app telemetry
- **SHAP explanations**: Per-prediction explainability for transparency and trust
- **Resampling**: Address class imbalance (High = 15.9%) with SMOTE or class-weight tuning
- **Temporal features**: If timestamps were available, session frequency and recency patterns could improve recall on High-addiction cases


