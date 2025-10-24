# Car Accident Severity Prediction

## Overview
This project predicts whether a road accident is **low/moderate (0)** or **severe (1)** using U.S. traffic data (2016–2023).  
The pipeline includes **EDA, feature engineering, PCA, SMOTE balancing, model tuning, threshold optimization, and SHAP explainability**.

**Objectives**:
- Identify key factors (mainly weather-focused factors) contributing to accident severity.
- Improve recall for severe accidents.
- Provide interpretable predictions for traffic safety planning.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Programming | Python 3.9+, Jupyter Notebook |
| Libraries | Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, SHAP |
| Techniques | EDA, Feature Engineering, PCA, Outlier Removal (IQR), SMOTE, Ensemble Learning, Hyperparameter Tuning |
| Visualization | Seaborn, Matplotlib |
| Explainability | SHAP (bar and beeswarm plots), Feature Importance |

---

## Dataset
- **Size:** 3 GB  
- **Sample Size:** 1,000,000 records  
- **Target Encoding:**  
  - Severity 1 & 2 → 0 (Low/Moderate)  
  - Severity 3 & 4 → 1 (Severe)  

**Features Used:**
- Location: `Start_Lat`, `Start_Lng`, `End_Lat`, `End_Lng`  
- Temporal: `Start_Time`, `End_Time`
- Weather: `Temperature(F)`, `Pressure(in)`, `Humidity(%)`, `Wind_Speed(mph)`, `Visibility(mi)`, `Precipitation(in)`, `Weather_Condition`
- Traffic: `Traffic_Signal`, `Crossing`, `Junction`

---

## Methodology

1. **Exploratory Data Analysis (EDA)**
   - Visualized severity distribution and feature histograms.
   - Removed outliers using IQR.

2. **Feature Engineering**
   - Extracted hour of day, day/night, and weather-related features.
   - Applied PCA for dimensionality reduction on categorical/weather features.

3. **Data Balancing**
   - Used SMOTE to handle class imbalance in training data.

4. **Model Training and Tuning**
   - Compared Logistic Regression, Random Forest, and XGBoost.
   - Performed hyperparameter tuning with GridSearchCV on XGBoost.

5. **Evaluation**
   - Metrics: Accuracy, F1, Precision-Recall (PR-AUC)
   - Threshold tuning to improve recall of severe accidents.
   - Model interpretation with SHAP plots.

---

## Results

| Model | Accuracy | F1 (Severe=1) | PR-AUC |
|-------|---------|----------------|--------|
| Logistic Regression | 0.64 | 0.46 | 0.35 |
| Random Forest | 0.83 | 0.56 | 0.57 |
| XGBoost (Base) | 0.84 | 0.57 | 0.62 |
| XGBoost (Tuned) | 0.85 | 0.58 | 0.64 |

- Severe accidents are the minority class (~20%).  
- XGBoost (Tuned) achieved the highest PR-AUC (0.64).  
- Threshold tuning (τ=0.35) improved severe class recall from 52% → 69%.  
- Most important predictive features: Latitude, Longitude, Temperature, Wind Chill.

---

## Visualizations

- Severity Distribution  
- Correlation Heatmap  
- Top 20 Feature Importances (Random Forest)  
- SHAP Summary Plot  
- Precision-Recall Curves (All Models)  
- Confusion Matrix (τ = 0.35)

---

## Threshold Optimization

- Evaluated precision, recall, and F1 score across multiple thresholds.  
- Selected **τ = 0.35** for the best balance between recall of severe accidents and overall precision.

---

## Explainability

- SHAP plots indicate higher temperature increases severity probability.  
- Specific geographic regions (latitude/longitude clusters) are higher risk.  
- Traffic signals and crossings help reduce severity risk.
