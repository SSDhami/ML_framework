from typing import List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import  MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import optuna
import logging

class Regressor:
    def __init__(self,) -> None:
        # Initialize Regressor class
        self.params = None
        self.logger = logging.getLogger(__name__)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # Train the model
        try:
            self.model = self.Regressor_type.fit(x_train,y_train)
            self.logger.info("Model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {e}")

    def predict(self, x_test: np.ndarray) -> None:
        # Make predictions
        try:
            self.prediction = self.model.predict(x_test)
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {e}")

    def score(self, x_test, y_test) -> Tuple[float, float, float]:
        
        try:
            # Calculate mean squared error
            self.mse_score = self.model.score(x_test,y_test)
            self.logger.info("Mean squared error calculated successfully.")

            # Calculate R2 score
            r2 = r2_score(y_test, self.model.predict(x_test))
            self.logger.info("R-squared score calculated successfully.")

            # Calculate MAE
            mae = mean_absolute_error(y_test, self.model.predict(x_test))
            self.logger.info("Mean Absolute Error calculated successfully.")

            return self.mse_score,r2, mae
        
        except Exception as e:
            self.logger.error(f"Error occurred while calculating mean squared error: {e}")


class Linear_Regressor(Regressor):

    def __init__(self) -> None:
        super().__init__()
        # Initialize Linear Regression classifier
        self.Regressor_type = LinearRegression(self.params)


class KNN_regressor(Regressor):

    def __init__(self) -> None:
        super().__init__()
        # Initialize KNN_regressor
        self.Regressor_type = KNeighborsRegressor(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'n_neighbors' : trial.suggest_int('n_neighbors',1,50),
                'weights' : trial.suggest_categorical('weights',['uniform', 'distance']),
                'p' : trial.suggest_int('p',1, 2)    
            }
            
            model_temp1 = KNeighborsRegressor(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params
        self.Regressor_type = KNeighborsRegressor(self.params)

class Decision_tree(Regressor):

    def __init__(self) -> None:
        super().__init__()
        # Initialize Decision Tree 
        self.Regressor_type = DecisionTreeRegressor(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'criterion' : trial.suggest_categorical('criterion',['squared_error', 'absolute_error','friedman_mse']),
                'splitter' : trial.suggest_categorical('splitter',['best', 'random'])  
            }
            
            model_temp1 = DecisionTreeRegressor(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params
        self.Regressor_type = DecisionTreeRegressor(self.params)

class Random_forest_regressor(Regressor):

    def __init__(self) -> None:
        super().__init__()
        self.Regressor_type = RandomForestRegressor(self.params)

    def hypertune_params(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        # Method for hyperparameter tuning
        def objective (trial):
            params = {
                'n_estimators' : trial.suggest_int('n_estimators',100,500,step = 100),
                'criterion' : trial.suggest_categorical('criterion',['squared_error', 'absolute_error', 'friedman_mse'])  
            }
            
            model_temp1 = RandomForestRegressor(**params).fit(x_train,y_train)
            score = model_temp1.score(x_test,y_test)
            return score


        study = optuna.create_study(direction = "maximize")

        study.optimize(objective, n_trials=100)

        self.params = study.best_params
        self.Regressor_type = RandomForestRegressor(self.params)

class SVC(Regressor):

    def __init__(self) -> None:
        super().__init__()
        self.Regressor_type = SVC(self.params)

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
        self.Regressor_type = SVC(self.params)

class ANN_sklearn_regressor(Regressor):

    def __init__(self, **params) -> None:
        super().__init__()
        # Initialize MLP classifier
        self.Regressor_type = MLPRegressor(**params)

class ANN_keras_regressor(Regressor):

    def __init__(self, **params) -> None:
        super().__init__()
        self.regressor = Sequential()
        self.logger = logging.getLogger(__name__)

    def add_layers(self, layer_units: List[int], activations: List[str]) -> None:
        try:
            for units, activation in zip(layer_units, activations):
                self.regressor.add(Dense(units=units, activation=activation))
            self.logger.info("Layers added successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while adding layers: {e}")

    def compile(self, **params) -> None:
        try:
            self.regressor.compile(**params)
            self.logger.info("Model compiled successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while compiling the model: {e}")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **params) -> None:
        try:
            self.regressor.fit(x_train,y_train,**params)
            self.logger.info("Model trained successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while training the model: {e}")

    def predict(self, x_test: np.ndarray) -> None:
        try:
            self.tf_reg_pred = self.regressor.predict(x_test).flatten()
            self.logger.info("Predictions made successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred while making predictions: {e}")

    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
        try:
            self.tf_reg_mse = mean_squared_error(y_test, self.tf_reg_pred)
            self.logger.info("Mean squared error calculated successfully.")

            # Calculate R2 score
            r2 = r2_score(y_test, self.tf_reg_pred)
            self.logger.info("R-squared score calculated successfully.")

            # Calculate MAE
            mae = mean_absolute_error(y_test, self.tf_reg_pred)
            self.logger.info("Mean Absolute Error calculated successfully.")

            return self.tf_reg_mse, r2, mae
        except Exception as e:
            self.logger.error(f"Error occurred while calculating scores: {e}")