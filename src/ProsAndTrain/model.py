##
# @file model.py
# @brief A wrapper class for various machine learning classification models.
#
# This module provides a unified interface for working with different classification algorithms
# including Random Forest, XGBoost, SVM, and CatBoost. It handles model initialization,
# training, prediction, and evaluation through a common API.
#

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier


##
# @class MLModel
# @brief A wrapper class that provides a unified interface for multiple classification models.
#
# This class allows instantiation of different classification models using a simple string parameter,
# and provides common methods for training, prediction and evaluation.
#
class MLModel:
   ##
   # @brief Initializes the ML model based on the model_name passed.
   #
   # @param model_name Name of the model to use. Available options: 'random_forest', 'xgboost', 'svm', 'catboost'.
   #
   def __init__(self, model_name: str):
       """
       Initializes the ML model based on the model_name passed.
       
       :param model_name: Name of the model to use. Available options: 'random_forest', 'xgboost', 'svm', 'catboost'.
       """
       self.model_name = model_name.lower()
       self.model = self._get_model()

   ##
   # @brief Returns the corresponding model based on the provided model_name.
   #
   # @return The instantiated machine learning model object.
   # @throws ValueError If the specified model name is not recognized.
   #
   def _get_model(self):
       """
       Returns the corresponding model based on the provided model_name.
       """
       if self.model_name == 'random_forest':
           return RandomForestClassifier()
       elif self.model_name == 'xgboost':
           return xgb.XGBClassifier()
       elif self.model_name == 'svm':
           return SVC()
       elif self.model_name == 'catboost':
           return CatBoostClassifier(silent=True)
       else:
           raise ValueError(f"Model '{self.model_name}' is not recognized. Available options: 'random_forest', 'xgboost', 'svm', 'catboost'.")

   ##
   # @brief Trains the model on the provided training data.
   #
   # @param X_train Features for training.
   # @param y_train Target values for training.
   #
   def train(self, X_train, y_train):
       """
       Trains the model on the provided training data.
       
       :param X_train: Features for training.
       :param y_train: Target values for training.
       """
       self.model.fit(X_train, y_train)

   ##
   # @brief Makes predictions on the provided test data.
   #
   # @param X_test Features for prediction.
   # @return Predicted class labels.
   #
   def predict(self, X_test):
       """
       Makes predictions on the provided test data.
       
       :param X_test: Features for prediction.
       :return: Predicted values.
       """
       return self.model.predict(X_test)

   ##
   # @brief Returns the current model instance.
   #
   # @return The ML model instance.
   #
   def get_model(self):
       """
       Returns the current model instance.
       
       :param X_test: Features for testing.
       :param y_test: Target values for testing.
       :return: Accuracy score.
       """
       return self.model

   ##
   # @brief Returns the accuracy score of the model on the provided test data.
   #
   # @param X_test Features for testing.
   # @param y_test Target values for testing.
   # @return Accuracy score between 0 and 1.
   #
   def get_score(self, X_test, y_test):
       """
       Returns the accuracy score of the model on the provided test data.
       
       :param X_test: Features for testing.
       :param y_test: Target values for testing.
       :return: Accuracy score.
       """
       return self.model.score(X_test, y_test)