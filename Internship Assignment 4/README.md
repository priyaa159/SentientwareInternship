# Loan Default Prediction using Machine Learning

This project aims to predict whether a loan application will be approved or not based on various applicant attributes. We use supervised machine learning algorithms and evaluate them based on performance metrics.

## Project Overview

- Dataset: Loan Prediction Dataset (from Kaggle)
- Goal: Predict loan approval status (Yes/No)
- Type: Binary Classification

## Tools and Technologies

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## Steps Followed

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical variables using label encoding
   - Normalized numerical features

2. **Exploratory Data Analysis**
   - Checked data distribution
   - Explored feature correlation with the target
   - Visualized class imbalance and patterns

3. **Model Training**
   - Trained three classification models:
     - Decision Tree Classifier
     - Random Forest Classifier
     - Gradient Boosting Classifier

4. **Model Evaluation**
   - Evaluated models using accuracy, precision, recall, and confusion matrix
   - Visualized confusion matrices and important features

5. **Hyperparameter Tuning (Bonus)**
   - Used GridSearchCV on Gradient Boosting to find the best parameters
   - Tuned values like number of estimators, learning rate, and max depth

## Results

The Gradient Boosting model performed best among all with the highest accuracy and balanced precision/recall. After hyperparameter tuning, it showed further improvement and better generalization on test data.

## Best Model

Gradient Boosting Classifier was selected as the best model based on:
- Strong predictive performance
- Balanced metric scores
- Clear feature importance insights
- Stability after tuning

## Top Features Contributing to Prediction

- Credit History
- Applicant Income
- Loan Amount
- Education
- Property Area

## How to Run

1. Clone the repository or download the project folder
2. Open the notebook in Jupyter or Google Colab
3. Install required libraries:
   - numpy, pandas, matplotlib, seaborn, scikit-learn
4. Run all cells in order to preprocess, train, and evaluate models
