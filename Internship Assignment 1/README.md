# Minor Project 1: Titanic Survival Prediction

## Project Overview
This project focuses on analyzing the Titanic dataset to predict passenger survival using machine learning techniques. The dataset contains information about passengers aboard the Titanic, including demographics, ticket details, and survival status. The goal is to build an effective classification model that predicts whether a passenger survived or not.

## Dataset
- Source: [Titanic Dataset from DataScienceDojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- Description: The dataset includes features such as passenger class, sex, age, fare, number of siblings/spouses aboard, number of parents/children aboard, embarkation port, and survival status.

## Features Used
- Passenger Class (`Pclass`)
- Sex (`Sex`)
- Age (`Age`)
- Fare (`Fare`)
- Number of Siblings/Spouses aboard (`SibSp`)
- Number of Parents/Children aboard (`Parch`)
- Embarkation Port (`Embarked`)

## Data Preprocessing
- Handling missing values (e.g., filling missing ages with median values)
- Encoding categorical variables (e.g., sex, embarkation port)
- Feature engineering (e.g., extracting titles from names)
- Scaling numerical features where applicable

## Exploratory Data Analysis (EDA)
- Summary statistics and visualizations of key features
- Survival rates by gender, class, age groups, and embarkation port
- Identification of important predictors of survival

## Model Development
- Algorithms used: Logistic Regression, Decision Trees, or others as applicable
- Model training, validation, and evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Key Findings
- Overall survival rate is approximately 38.4%
- Female passengers had a higher survival rate (~74%) compared to males (~19%)
- First-class passengers had the highest survival rate (~63%)
- Children had higher survival probability
- Embarkation port also influences survival chances

## How to Run
1. Clone this repository:
   ```bash
   https://github.com/priyaa159/SentientwareInternship.git
