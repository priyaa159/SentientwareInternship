# 🩺 Breast Cancer Classification using Logistic Regression

This project focuses on building a **Logistic Regression** model to classify whether a tumor is **benign** or **malignant** using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.

## 📌 Project Overview

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Model:** Logistic Regression
- **Problem Type:** Binary Classification (Benign vs Malignant)
- **Libraries:** NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report

## 📊 Exploratory Data Analysis (EDA)

Performed data cleaning and visualization:
- Checked for missing/null values
- Visualized class distribution (Benign vs Malignant)
- Heatmap of feature correlation
- Count plots and histograms for key features

## 🔧 Features Used

The dataset contains 30 numeric features related to cell nuclei from digitized images:
- `mean radius`, `mean texture`, `mean perimeter`, etc.
- Diagnosis is the target column:  
  - `M` = Malignant  
  - `B` = Benign

## 🧠 Model Building

- **Train-Test Split:** 80% training, 20% testing
- **Model:** `LogisticRegression()` from Scikit-learn
- Trained on scaled/normalized data for better performance.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
📈 Model Evaluation
Evaluated model performance using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

Visualizations include:

Heatmap of confusion matrix

Bar chart of accuracy

Predicted vs Actual comparison

📝 Summary
The logistic regression model performs well in classifying tumors based on diagnostic features. It achieves high accuracy and shows good generalization on test data.

🚀 How to Run
Clone the repository or open the notebook in Google Colab.

Install dependencies (if running locally):

bash
Copy
Edit
pip install numpy pandas seaborn matplotlib scikit-learn
Run all cells in the notebook to view the results.