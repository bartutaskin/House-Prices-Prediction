# 🏠 House Prices Prediction with XGBoost & Optuna

## Overview
This project aims to predict the sale prices of houses in Ames, Iowa, using machine learning techniques. The dataset contains various features related to the properties, including construction details, quality assessments, and location information. The goal is to build a high-performing regression model that can estimate house prices with minimal error.

The model is built using **XGBoost**, a powerful and efficient implementation of gradient boosting. To enhance performance, **Optuna** is used for automated hyperparameter tuning, which led to a final cross-validated RMSE of **0.12757**.


## Data
The project uses the **Ames Housing Dataset**, which is widely used in Kaggle competitions. The dataset consists of both numerical and categorical features describing properties in Ames, Iowa. It includes the target variable `SalePrice`, which represents the sale price of each property.

### Data Files:
- **train.csv**: Contains the training data, including both the features and the target variable (`SalePrice`).
- **test.csv**: Contains the test data, used for generating predictions. The target variable `SalePrice` is not included.

## Approach

### 1. Data Preprocessing
The preprocessing steps include:
- Handling missing values by imputing them with suitable strategies such as filling with the mode for categorical variables and filling with the mean for numerical variables.
- Feature engineering to create new features that could improve model performance (e.g., total square footage, number of bathrooms).
- Encoding categorical features using ordinal encoding for features with a specific order (e.g., quality levels) and one-hot encoding for other categorical variables.
- Handling outliers by removing extreme values that could negatively impact the model.

### 2. Model Training
The base model is trained using XGBoostRegressor. A 5-fold cross-validation is performed to evaluate the baseline performance.

### 3. Evaluation
The model’s performance is evaluated using Root Mean Squared Error (RMSE) during cross-validation. The final model achieved a RMSE of 0.12757, indicating strong predictive performance.

### 4. Submission
The final model is used to predict house prices on the test set. Predictions are saved in data/predictions2.csv in the required format:
```bash
Id,SalePrice
1461,123456.78
1462,234567.89
...
```

## Result
The final XGBoost model achieved a **cross-validated RMSE of 0.12757**. This means that, on average, the predicted sale prices differ from the actual prices by approximately 12.8% in root mean square error terms.



