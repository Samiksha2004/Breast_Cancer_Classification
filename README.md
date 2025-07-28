
# Breast Cancer Classification using Machine Learning

This project uses machine learning to classify whether a tumor is **malignant** or **benign** using the **Breast Cancer Wisconsin Diagnostic Dataset**. It demonstrates a full ML workflow from data cleaning and visualization to model building and evaluation — all in a business-critical healthcare context.The goal is not just accuracy, but also **model interpretability, generalizability**, and **comparative insights** — making it ideal for decision-making scenarios like healthcare risk assessment.

---

## Problem Statement

Breast cancer is one of the most common forms of cancer among women worldwide. Early and accurate detection is critical for effective treatment. The goal of this project is to build a **classification model** that predicts whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on diagnostic features.

---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Features:** 30 numeric features such as radius, texture, perimeter, area, smoothness, etc.
- **Target:** Diagnosis (Malignant = 1, Benign = 0)

---

## Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

---

## Workflow

1. **Data Loading & Exploration**
   - Shape, info, null values, class distribution
2. **EDA (Exploratory Data Analysis)**
   - Correlation matrix
   - Feature distribution plots
   - Class imbalance check
3. **Data Preprocessing**
   - Label encoding
   - Feature scaling (StandardScaler)
   - Train-test split
4. **Model Building**
   - Logistic Regression
   - Random Forest
   - SVM (Support Vector Machine)
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix
   - ROC-AUC Curve
6. **Feature Importance**
   - Using Random Forest
7. **(Optional) Explainability**
   - SHAP or LIME (future enhancement)

---

## 📊 Model Performance Summary

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|----------------|
| Logistic Regression| 94.9%          | **92.98%**     |
| SVM                | 92.5%          | 90.35%         |
| Random Forest      | **100%**       | **94.74%**     |

Random Forest achieved the highest test accuracy but showed signs of **overfitting** due to perfect training accuracy.  
Logistic Regression performed very well and remained highly interpretable.  
SVM, though slightly lower in accuracy, was tested to evaluate the impact of **non-linear decision boundaries**.


## Key Learnings
Evaluating multiple models helps identify the **trade-off between performance and explainability**.
- **Random Forests** offer high accuracy but may overfit on smaller datasets.
- **Logistic Regression** is a strong baseline model, especially in domains like healthcare.
- **SVMs** are useful to detect **non-linear relationships**, even if not always more accurate.

---

## Key Takeaways

- Feature scaling significantly improved model performance
- Strong correlation between tumor radius and malignancy
- Random Forest performed best with minimal tuning

