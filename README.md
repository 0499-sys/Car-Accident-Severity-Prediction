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
   - Metrics: Accuracy, F1-Score, Precision-Recall (PR-AUC)
   - Threshold tuning to improve recall of severe accidents.

---

## Results

| Model | Accuracy | F1 (Severe=1) | PR-AUC |
|-------|---------|----------------|--------|
| Logistic Regression | 0.64 | 0.46 | 0.35 |
| Random Forest | 0.83 | 0.56 | 0.57 |
| XGBoost (Base) | 0.84 | 0.57 | 0.62 |
| XGBoost (Tuned) | 0.85 | 0.58 | 0.64 |

**XGBoost (Tuned)** achieved the **highest PR-AUC (0.64)** and the best balance between recall and precision.  
After **threshold tuning (τ = 0.35)**, **recall improved from 52% → 69%**, helping identify more severe cases.

<img width="784" height="384" alt="output_37_2" src="https://github.com/user-attachments/assets/851fdd14-1abc-47fa-b61c-64ac65033e64" />

---

## Threshold Optimization  

- **Goal:** Improve recall of severe accidents without too many false alarms.  
- Best performance at **τ = 0.35**:  
  - Precision ≈ 0.54  
  - Recall ≈ 0.69  
  - F1 ≈ 0.60  

---

## Confusion Matrices  

**Default (Threshold = 0.5)**  
<img width="330" height="257" alt="output_37_4" src="https://github.com/user-attachments/assets/05f4e230-4a20-48ef-b97d-67affe44c4a6" />

**Optimized (Threshold = 0.35)**  
<img width="330" height="257" alt="output_40_1" src="https://github.com/user-attachments/assets/4b50c761-c666-4645-b463-cf9d350842fc" />

The tuned threshold increases detection of severe accidents (**True Positives**) while slightly raising false positives.  
This is acceptable because **missing severe cases is more critical** than issuing false alerts.

---

## Why Lower Recall Isn’t Always Bad  

Although recall can be pushed higher, **very high recall (>80%)** drastically lowers **precision**, causing too many false alarms.  
In a real-world traffic system, over-predicting severity wastes emergency resources.  

**Balanced recall (~69%)** ensures a **realistic, actionable model**.  

---

## Model Explainability (SHAP Analysis)  

**Feature Importance**  
<img width="784" height="384" alt="output_27_0" src="https://github.com/user-attachments/assets/bb0640e9-0493-4f69-b9fd-7bee80534593" />

**SHAP Summary (Impact by Value)**  
<img width="562" height="734" alt="output_31_0" src="https://github.com/user-attachments/assets/a4f9fef3-4921-46ed-9718-7b4df9744f8f" />  
<img width="584" height="734" alt="output_30_0" src="https://github.com/user-attachments/assets/216e16da-3149-4d13-beba-4a9b7b56dcb9" />

### Key Insights  
- **Temperature (F)**, **Latitude/Longitude**, and **Wind Chill (F)** are top predictors.  
- **Higher temperatures** slightly increase accident severity probability.  
- **Specific location clusters** (by Start_Lat/Start_Lng) correspond to historically severe regions.  
- **Traffic signals** and **crossings** correlate with **lower severity**, indicating safer zones.  
- **Wind speed** and **distance** also contribute — long, windy roads tend to increase risk.
