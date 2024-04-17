
import re
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import optuna


class Classifier:
    def __init__(self) -> None:
        self.params = None
        self.logger = logging.getLogger(__name__)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # Train the model
        try:
            self.model = self.classifier_type.fit(x_train, y_train)
            self.logger.info("Model fitted successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while fitting the model: {e}")

    def predict(self, x_test: np.ndarray) -> None:
        try:
            self.prediction = self.model.predict(x_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while making predictions: {e}")

    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float]:
        try:
            # Calculate accuracy score
            self.accuracy_score = accuracy_score(y_test, self.prediction)
            self.logger.info("Accuracy score calculated successfully.")

            # Calculate precision score
            self.precision = precision_score(y_test, self.prediction, average='weighted')
            self.logger.info("Precision score calculated successfully.")

            # Calculate recall score
            self.recall = recall_score(y_test, self.prediction, average='weighted')
            self.logger.info("Recall score calculated successfully.")

            # Calculate F1-score
            self.f1 = f1_score(y_test, self.prediction, average='weighted')
            self.logger.info("F1-score calculated successfully.")

            return self.accuracy_score, self.precision, self.recall, self.f1
        except Exception as e:
            self.logger.error(f"An error occurred while calculating scores: {e}")

class Logistic_Regression(Classifier):

    def __init__(self) -> None:
        super().__init__()
        # Initialize logistic regression classifier
        self.classifier_type = LogisticRegression(self.params)

class KNN_classifier(Classifier):

    def __init__(self) -> None:
        super().__init__()
        # Initialize KNN classifier
        self.classifier_type = KNeighborsClassifier(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'n_neighbors' : trial.suggest_int('n_neighbors',1,50),
                'weights' : trial.suggest_categorical('weights',['uniform', 'distance']),
                'p' : trial.suggest_int('p',1, 2)    
            }
            
            model_temp1 = KNeighborsClassifier(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params['n_neighbors'] , study.best_params['weights'], study.best_params['p']


        self.classifier_type = KNeighborsClassifier(self.params)

class Decision_tree(Classifier):

    def __init__(self) -> None:
        super().__init__()
        # Initialize Decision Tree classifier
        self.classifier_type = DecisionTreeClassifier(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'criterion' : trial.suggest_categorical('criterion',['gini', 'entropy']),
                'splitter' : trial.suggest_categorical('splitter',['best', 'random'])  
            }
            
            model_temp1 = DecisionTreeClassifier(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params
        self.classifier_type = DecisionTreeClassifier(self.params)

class Random_forest(Classifier):

    def __init__(self) -> None:
        super().__init__()
        # Initialize Random Forest classifier
        self.classifier_type = RandomForestClassifier(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'n_estimators' : trial.suggest_int('n_estimators',100,500,step = 100),
                'criterion' : trial.suggest_categorical('criterion',['gini', 'entropy'])   
            }
            
            model_temp1 = RandomForestClassifier(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=20)

        self.params = study.best_params
        self.classifier_type = RandomForestClassifier(self.params)

class SVC(Classifier):

    def __init__(self) -> None:
        super().__init__()
        # Initialize SVC classifier
        self.classifier_type = SVC(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'kernel' : trial.suggest_categorical('kernel',['linear', 'poly', 'rbf', 'sigmoid']),
                'C' : trial.suggest_int('C',1, 10)    
            }
            
            model_temp1 = SVC(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params
        self.classifier_type = SVC(self.params)

class ANN_sklearn(Classifier):

    def __init__(self, **params) -> None:
        super().__init__()
        # Initialize MLP classifier
        self.classifier_type = MLPClassifier(**params)

class ANN_keras(Classifier):

    def __init__(self, **params) -> None:
        super().__init__()
        # Initialize Sequential model
        self.classifier = Sequential()

    def add_layers(self, layer_units: List[int], activations: List[str]) -> None:
       # Method to add layers to the neural network
       for units, activation in zip(layer_units, activations):
            self.classifier.add(Dense(units=units, activation=activation))

    def compile(self, **params) -> None:
        self.classifier.compile(**params)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **params) -> None:
        try:
            self.classifier.fit(x_train, y_train, **params)
            self.logger.info("Model fitted successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while fitting the model: {e}")

    def predict(self, x_test: np.ndarray) -> None:
        try:
            self.tf_class_pred = np.argmax(self.classifier.predict(x_test), axis=1)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred while making predictions: {e}")

    #def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
     #   try:
      #      self.accuracy_score = accuracy_score(y_test, self.tf_class_pred)
       #     self.logger.info("Accuracy score calculated successfully.")
        #    return self.accuracy_score
        #except Exception as e:
         #   self.logger.error(f"An error occurred while calculating accuracy score: {e}")