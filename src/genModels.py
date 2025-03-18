"""
Trains and saves a machine learning model for cervical cancer risk prediction.

This script loads cervical cancer risk factor data, processes it, and trains a 
selected machine learning model (XGBoost by default). The model is evaluated on 
both training and test sets, and then saved as a pickle file for later use.

The script performs the following operations:
- Loads cervical cancer risk factor dataset
- Processes the data using a custom processing function
- Selects a machine learning model (XGBoost)
- Splits the data into training and test sets
- Applies feature scaling using StandardScaler
- Handles class imbalance using SMOTE oversampling
- Trains the model on the resampled data
- Evaluates the model performance on both training and test sets
- Saves the trained model to disk

Note: The target variable 'Biopsy' is used as the prediction target

Dependencies:
- ProAndTrain.dataProcessing: Module for data preprocessing
- ProAndTrain.model: Module containing the MLModel class
- numpy: For numerical operations
- pandas: For data manipulation
- sklearn: For preprocessing and model evaluation
- imblearn: For handling class imbalance
- joblib: For saving the trained model
"""
import numpy as np
import pandas as pd

from ProsAndTrain import dataProcessing, model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE

import joblib


df = pd.read_csv("../data/risk_factors_cervical_cancer.csv")
df = dataProcessing.process_data(df)

ChoosenModel = "xgboost"
Model = model.MLModel(ChoosenModel)

X = np.array(df.drop(columns = ['Biopsy'])).astype('float32')
y = np.array(df['Biopsy']).astype('float32')

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, x_val, y_test, y_val = train_test_split(X, y, test_size = 0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

Model.train(X_resampled, y_resampled)

result_train = Model.get_score(X_train, y_train)
result_test = Model.get_score(X_test, y_test)
print(result_train, result_test)

joblib.dump(Model.get_model(), "../genModels/" + ChoosenModel + ".pkl")
