# 🚗 Car Accident Severity Prediction (2025)

> Machine Learning pipeline predicting the **severity of road accidents** using 1M+ U.S. traffic records.  
> Combines **EDA, feature engineering, PCA compression, SMOTE balancing, model tuning, and SHAP explainability**.

---

## 🧠 Project Overview

This project predicts whether an accident is **mild/moderate (0)** or **severe (1)** using weather, time, and location data.  
It uses the **US_Accidents** dataset (2016–2023) and delivers interpretable, production-ready results through model comparison, hyperparameter optimization, and threshold tuning.

**Objectives**
- Identify key factors that contribute to accident severity.  
- Improve recall for severe cases using threshold tuning.  
- Provide interpretable model outputs for safety & traffic planning.

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|-------|
| **Programming** | Python 3.9+, Jupyter Notebook |
| **Libraries** | Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, SHAP, Matplotlib, Seaborn |
| **Techniques** | EDA, Feature Engineering, PCA, Outlier Removal (IQR), SMOTE, Ensemble Learning, Model Tuning |
| **Visualization** | Seaborn, Matplotlib |
| **Explainability** | SHAP (bar + beeswarm), Feature Importance |

---

## 📂 Dataset

- **Source:** [US_Accidents (Kaggle / Open Data)](https://www.kaggle.com/sobhanmoosavi/us-accidents)
- **Records Used:** 1,000,000 stratified samples  
- **Target Encoding:**  
  `Severity 1 & 2 → 0 (Low/Moderate)`  
  `Severity 3 & 4 → 1 (Severe)`  

**Selected Features**
- Location: `Start_Lat`, `Start_Lng`, `End_Lat`, `End_Lng`  
- Temporal: `Start_Time`, `End_Time`, Twilight Indicators  
- Weather: `Temperature(F)`, `Pressure(in)`, `Humidity(%)`, `Wind_Speed(mph)`  
- Traffic Indicators: `Traffic_Signal`, `Crossing`, `Junction`

---

## 🧩 Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Visualized severity distribution and feature histograms  
   - Removed outliers using IQR across numeric columns  

2. **Feature Engineering**  
   - Extracted hour of day, day/night features  
   - Encoded wind and weather conditions → **PCA(13)** for dimensionality reduction  

3. **Data Balancing**  
   - Applied **SMOTE** on training data to handle imbalance  

4. **Model Training**  
   - Compared Logistic Regression, Random Forest, and XGBoost  
   - Tuned XGBoost hyperparameters using `GridSearchCV`  

5. **Evaluation**  
   - Metrics: Accuracy, F1, Precision-Recall Curve (PR-AUC)  
   - Threshold tuning to improve recall for severe cases  
   - Model interpretation using SHAP  

---

## 📊 Results Overview

| Model | Accuracy | F1 (Severe=1) | PR-AUC |
|:------|:---------:|:-------------:|:------:|
| Logistic Regression | 0.64 | 0.46 | 0.35 |
| Random Forest | 0.83 | 0.56 | 0.57 |
| XGBoost (Base) | 0.84 | 0.57 | 0.62 |
| **XGBoost (Tuned)** | **0.85** | **0.58** | **0.64** |

**Key Insights**
- Most accidents are of **Severity 2 (≈80%)**, leading to class imbalance.  
- After balancing, the tuned XGBoost achieved the **best PR-AUC (0.64)**.  
- **Threshold tuning (τ=0.35)** improved recall of severe cases from 52% → 69%.  
- **Top predictive factors:** Latitude, Longitude, Pressure, Temperature, Humidity, and Wind Chill.

---

## 📈 Visual Results

### Severity Distribution
![Severity Distribution](docs/severity_proportions.png)

### Correlation Heatmap
![Correlation Heatmap](docs/corr_heatmap.png)

### Top 20 Feature Importances (Random Forest)
![Feature Importances](docs/rf_feature_importances.png)

### SHAP Summary Plot
![SHAP Plot](docs/shap_beeswarm.png)

### Precision-Recall Curves (All Models)
![PR Curves](docs/pr_all_models.png)

### Confusion Matrix (τ = 0.35)
![Confusion Matrix](docs/cm_xgb_tau035.png)

---

## 🧮 Threshold Optimization
Balancing **precision vs recall** for severe accidents:

![Threshold Tuning](docs/threshold_sweep.png)

---

## 💡 Explainability

**SHAP results reveal:**
- Higher temperatures, pressure, and humidity increase accident severity probability.  
- Latitude/Longitude clusters represent geographically high-risk regions.  
- Traffic signals and crossings play a notable preventive role.
