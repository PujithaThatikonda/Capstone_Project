# Module 2 — Analytics Pipeline

## 1. Project Overview

This project implements a complete Machine Learning and Data Analytics pipeline using the Titanic dataset.

The workflow follows a real-world data science process:

Load Data → Clean Data → Explore Data → Engineer Features →
Train Models → Evaluate Models → Tune Models →
Handle Imbalanced Data → Save Final Model

The dataset used is the Titanic passenger dataset available through Seaborn.

Target Variable:

survived

0 = Passenger Did Not Survive

1 = Passenger Survived

---

## 2. Tools and Libraries Used

The project uses:

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- joblib

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

---

## 3. Dataset Description

Dataset:

Titanic Dataset

Features used include:

- survived
- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

Dataset Shape:

891 rows × 15 columns

Target Column:

survived

---

## 4. Data Profiling

Initial profiling was performed using:

df.info()
df.describe()
df.shape

The following were analyzed:

- Number of rows
- Number of columns
- Data types
- Missing values
- Summary statistics

Missing value percentages were calculated for every column.

---

## 5. Data Cleaning

### Missing Value Strategy

The following rules were applied:

Missing < 5% - Drop rows

Missing between 5% and 30% - Impute values

Missing > 30% - Drop column or treat separately

### Numerical Features

Missing values replaced using:

Median Imputation

Examples:

- age
- fare

### Categorical Features

Missing values replaced using:

Mode Imputation

Examples:

- embarked

---

## 6. Exploratory Data Analysis (EDA)

EDA was performed to understand passenger characteristics and survival patterns.

### Univariate Analysis

Performed for:

- Age
- Fare

Visualizations:

- Histogram
- Box Plot

Statistics calculated:

- Mean
- Median
- Mode
- Standard Deviation
- Skewness

Outliers were identified using:

IQR Method

---

### Bivariate Analysis

Survival rate was analyzed across:

1. Gender

survival vs sex

2. Passenger Class

survival vs pclass

3. Gender + Passenger Class

survival vs sex and pclass

Visualizations:

- Count plots
- Bar charts
- Grouped survival charts

---

## 7. Correlation Analysis

Correlation matrix was generated using:

- survived
- pclass
- age
- sibsp
- parch
- fare

Heatmap was created using Seaborn.

The strongest positive and negative relationships were identified and interpreted.

---

## 8. Feature Scaling

Numerical columns:

- age
- fare

were standardized using:

StandardScaler()

Comparison was shown:

Before Scaling

After Scaling

This ensures features have comparable scales before model training.

---

## 9. Machine Learning Pipeline

The dataset was split into:

Training Set = 80%

Testing Set = 20%

Stratified sampling was used to preserve class distribution.

A preprocessing pipeline was built using:

ColumnTransformer

Components:

Numerical Features

- Median Imputer
- StandardScaler

Categorical Features

- Mode Imputer
- One-Hot Encoding

This preprocessing pipeline was integrated directly with machine learning models.

---

## 10. Classification Models

Three classification models were trained.

### Model 1 — Logistic Regression

Used as baseline model.

Advantages:

- Fast
- Interpretable
- Good baseline

---

### Model 2 — Decision Tree

Advantages:

- Easy to visualize
- Handles nonlinear relationships

Tree structure visualized using:

plot_tree()

---

### Model 3 — Random Forest

Advantages:

- Ensemble learning
- Better generalization
- Reduced overfitting

Random Forest achieved the best overall performance.

---

## 11. Model Evaluation

The following metrics were calculated for all models:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Additional evaluation:

Confusion Matrix

ROC Curve

Classification Report

Models were compared side-by-side.

---

## 12. Handling Imbalanced Data

Class distribution was analyzed before training.

Three approaches were compared:

### Baseline Model

Original data

### Class Weight Balancing

class_weight="balanced"

### SMOTE Oversampling

Synthetic Minority Over-sampling Technique

SMOTE was applied only on training data to avoid data leakage.

Performance differences were analyzed.

---

## 13. Hyperparameter Tuning

GridSearchCV was used on Random Forest.

Parameters tuned:

- n_estimators
- max_depth
- max_features

Best model selected using cross-validation performance.

Out-of-Bag Score (OOB) was also reported.

---

## 14. Regression Task

In addition to classification, a regression model was built.

Target:

fare

Algorithm:

Linear Regression

Metrics:

- MAE
- RMSE
- R² Score
- Adjusted R²

Residual plots were generated to study prediction errors.

---

## 15. Model Persistence

The final trained pipeline was saved using:

joblib.dump()

Saved File:

model_pipeline.pkl

Reloading:

joblib.load()

This allows the model to be reused without retraining.

---

## 16. Project Workflow

Titanic Dataset
      │
      ▼
Data Cleaning
      │
      ▼
EDA & Visualization
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Saving

---

## 17. Project Files

analytics/
│
├── cleaned_titanic.csv
├── 02_modeling.py
├── 01_eda.py
├── model_pipeline.pkl
├── titanic.csv
└── README.md

---

## 18. Key Findings

- Female passengers had higher survival rates.
- First-class passengers survived more frequently.
- Passenger class strongly influenced survival.
- Random Forest produced the highest classification performance.
- Feature engineering and preprocessing significantly improved results.

---

## 19. Learning Outcomes

Through this project I learned:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Classification Models
- Regression Models
- Hyperparameter Tuning
- Model Evaluation
- Pipeline Construction
- Model Serialization

---

## 20. Final Results

Successfully implemented:

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Correlation Analysis

✔ Feature Scaling

✔ Logistic Regression

✔ Decision Tree

✔ Random Forest

✔ SMOTE

✔ Hyperparameter Tuning

✔ Regression Analysis

✔ Model Persistence

This completes the Analytics Pipeline module.