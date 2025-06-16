# 🏡 House Price Prediction using Linear Regression

This project aims to predict **California housing prices** using a **Linear Regression** model. The project is built on top of the classic California Housing Dataset and includes exploratory data analysis, model training, evaluation, and visualization.

## 📌 Project Overview

- **Dataset Used:** California Housing Dataset (from `sklearn.datasets`)
- **Model Used:** Linear Regression
- **Evaluation Metrics:** Mean Squared Error (MSE), R² Score
- **Tools & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

## 📊 Exploratory Data Analysis (EDA)

A brief exploratory analysis was conducted with:
- Correlation heatmap
- Distribution of housing median prices
- Scatter plots between features and target

## 🔧 Features Used

Multiple sets of features were experimented with. The final model was trained with **8 features**:
- `MedInc` (median income)
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

## 🧠 Model Training

- Split the data into training and testing sets (80/20 split).
- Trained a Linear Regression model using `sklearn.linear_model.LinearRegression`.

## 📈 Model Evaluation

- **Mean Squared Error (MSE):** Low value indicates good performance.
- **R² Score:** Indicates how well the model captures the variance in the target variable.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
📉 Results Visualization
A plot comparing actual vs predicted prices shows model performance visually.

Helps identify underfitting/overfitting patterns.

📝 Summary
This project demonstrates how a basic regression model can predict housing prices with reasonable accuracy. It also highlights the importance of feature selection and EDA in improving model performance.

🚀 How to Run
Clone the repository or open the Colab Notebook.

Install required dependencies (if running locally).

Execute all cells to train and test the model.