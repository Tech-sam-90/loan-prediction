# Loan Status Prediction Using Machine Learning

**Author:** Samuel Adeniji

## Overview

This project implements a comprehensive machine learning pipeline for predicting loan approval status. It demonstrates end-to-end data science workflow including data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and interpretability analysis using SHAP (SHapley Additive exPlanations).

## Table of Contents

- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Models Implemented](#models-implemented)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)

## Dataset

**Source:** `Loan Status Prediction.csv`

The dataset contains information about loan applicants with the following characteristics:
- **Numerical Features:** Applicant Income, Coapplicant Income, Loan Amount
- **Categorical Features:** Gender, Married, Education, Self-Employed, Property Area, Dependents, Credit History, Loan Amount Term
- **Target Variable:** Loan Status (Approved/Not Approved)

## Project Objectives

1. Apply data cleaning, feature engineering, and encoding techniques
2. Train and evaluate multiple classification models
3. Interpret results using performance metrics and explainability tools
4. Reflect on ethical implications and bias detection in modeling

## Methodology

### Task 1: Exploratory Data Analysis (EDA)
- **Data Structure Analysis:** Identified categorical and numerical variables
- **Distribution Analysis:** 
  - Calculated skewness and kurtosis for numerical features
  - Generated histograms and KDE plots for distribution visualization
  - Analyzed evidence of non-normal distributions
- **Key Insights:** Identified skewed variables requiring transformation for better modeling performance

### Task 2: Data Cleaning and Outlier Treatment
- **Missing Value Handling:**
  - Imputed numerical variables (Applicant Income, Coapplicant Income, Loan Amount) with median values
  - Filled categorical variables with mode
- **Outlier Detection:**
  - Used IQR (Interquartile Range) method for outlier identification
  - Removed outliers from Applicant Income, Coapplicant Income, and Loan Amount
- **Visualization:** Generated boxplots before and after outlier treatment
- **Impact Analysis:** Discussed potential biases introduced by data cleaning decisions

### Task 3: Feature Engineering and Preprocessing
- **New Feature Creation:**
  - `TotalIncome`: Sum of Applicant Income and Coapplicant Income
  - `LoanPerIncome`: Loan amount to income ratio
- **Data Transformation:**
  - Applied log transformation to skewed numerical variables
  - Converted discrete numerical variables to categorical types
- **Encoding:**
  - **Label Encoding:** Used for ordinal variables (Loan Status, Dependents, Loan Amount Term)
  - **One-Hot Encoding:** Applied to nominal variables (Gender, Married, Education, Self-Employed, Property Area)
- **Scaling:** 
  - Implemented Robust Scaling for numerical features (resistant to outliers)
- **Feature Selection:** 
  - Dropped irrelevant features (Loan_ID)
  - Removed redundant features after creating engineered features
- **Correlation Analysis:** Generated annotated heatmap to examine feature relationships

### Task 4: Model Training and Evaluation
Implemented three classification algorithms with class balancing:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Stochastic Gradient Descent (SGD) Classifier**

**Evaluation Strategy:**
- 70-30 train-test split
- 5-fold cross-validation
- Class weighting to handle imbalanced data

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Task 5: Model Interpretability and Bias Detection
- **SHAP Analysis:**
  - Global feature importance visualization
  - Local explanations using waterfall plots
  - Instance-level prediction interpretation
- **Bias Detection:**
  - Analyzed SHAP contributions across demographic groups
  - Examined disparities in model predictions for different education levels
  - Assessed potential discrimination patterns

## Key Features

- **Comprehensive EDA:** Statistical analysis with skewness, kurtosis, and distribution plots
- **Robust Preprocessing:** Handling missing values, outliers, and feature scaling
- **Feature Engineering:** Created meaningful derived features
- **Multiple Models:** Comparison of different classification algorithms
- **Cross-Validation:** Rigorous model evaluation with 5-fold CV
- **Interpretability:** SHAP-based explainability for transparency
- **Bias Analysis:** Ethical considerations and fairness evaluation

## Models Implemented

### 1. Logistic Regression
- Maximum iterations: 1000
- Class weight: Balanced
- Best performing model selected for SHAP analysis

### 2. Support Vector Machine (SVM)
- Probability estimates enabled
- Class weight: Balanced

### 3. SGD Classifier
- Maximum iterations: 1000
- Tolerance: 1e-3
- Class weight: Balanced

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
shap
```

## Usage

1. **Install Dependencies:**
   ```python
   pip install numpy pandas matplotlib seaborn scipy scikit-learn shap
   ```

2. **Load Dataset:**
   Ensure `Loan Status Prediction.csv` is in the same directory as the notebook

3. **Run Notebook:**
   Execute cells sequentially in `loan_prediction.ipynb`

4. **Workflow:**
   - EDA and distribution analysis
   - Data cleaning and outlier removal
   - Feature engineering and encoding
   - Model training and evaluation
   - Interpretability analysis

## Results

### Model Performance
All models evaluated using cross-validation with detailed classification reports including:
- Accuracy scores with standard deviation
- Precision, Recall, F1-Score, and ROC-AUC metrics
- Confusion matrices for error analysis

### Feature Importance
SHAP analysis revealed:
- Most influential features for loan approval predictions
- Direction and magnitude of feature impacts
- Potential bias patterns across demographic groups

### Key Findings
- Log transformation improved distribution normality
- Feature engineering enhanced model performance
- Class balancing addressed dataset imbalance
- SHAP provided actionable insights into prediction drivers

## Ethical Considerations

### Bias Analysis
- Examined SHAP contributions across education levels
- Identified potential disparities in model predictions
- Discussed implications of using demographic features

### Fairness Concerns
- Gender and marital status as protected attributes
- Risk of perpetuating historical lending biases
- Importance of monitoring model fairness in production

### Recommendations
- Regular bias audits on model predictions
- Consider fairness constraints during training
- Human oversight for final lending decisions
- Transparent communication of model limitations

## Conclusion

This project demonstrates a complete machine learning workflow for loan prediction with emphasis on:
- Rigorous data preprocessing and feature engineering
- Multiple model comparison with robust evaluation
- Interpretability and explainability for stakeholder trust
- Ethical considerations and bias detection

The comprehensive approach ensures not only predictive accuracy but also fairness and transparency in automated lending decisions.

## Files

- `loan_prediction.ipynb`: Main Jupyter notebook with complete analysis
- `Loan Status Prediction.csv`: Dataset (not included in repository)
- `README.md`: This file

## License

This project is completed as part of an academic assignment.

---

**Note:** This project is for educational purposes. Any deployment in real-world lending scenarios should undergo thorough ethical review and regulatory compliance checks.
