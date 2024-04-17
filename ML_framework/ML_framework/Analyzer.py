import logging
import pandas as pd
import numpy as np
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from typing import List, Tuple

class Analyzer:
    def __init__(self, path: str) -> None:

        # Initialize Analyzer object with the path to the dataset
        self.path = path
        self.logger = logging.getLogger(__name__)

    def read_dataset (self) -> None:

        # Read the dataset based on its file format (CSV or ARFF)
        if self.path.endswith('.csv'):
            self.data_frame = pd.read_csv(self.path)
            self.df = self.data_frame
        elif self.path.endswith('.arff'):
            data, _ = arff.loadarff(self.path)
            self.data_frame = pd.DataFrame(data)
            self.df = self.data_frame
        else:
            self.logger.error("Unsupported file format. Please provide either CSV or ARFF file.")

    def describe(self) -> pd.DataFrame:

        # Generate descriptive statistics of the dataset
        return self.df.describe()

    def drop_missing_data(self) -> None:

        # Drop rows with missing values
        self.df = self.df.dropna()
        self.logger.info("Missing data dropped successfully.")
     
    def drop_columns(self, list_features: List[str]) -> None:

        # Drop specified columns from the dataset
        self.df= self.df.drop(list_features, axis = 1)
        self.logger.info({list_features}," dropped successfully.")
    
    def encode_features_categorical(self, categorical_columns: List[str]) -> None:

        # Perform one-hot encoding for categorical features
        self.df = pd.get_dummies(self.df,columns = categorical_columns,dtype=float)
        self.logger.info("categorical features Encoded successfully.")
        
    def encode_features_ordinal(self,ordinal_columns: List[str]) -> None:

        # Perform ordinal encoding for ordinal features
        self.df[ordinal_columns]=OrdinalEncoder().fit_transform(self.df[ordinal_columns].values)
        self.logger.info("ordinal features Encoded successfully.")

    def encode_features_nominal(self,nominal_columns: List[str]) -> None:

        # Perform nominal encoding for nominal features
        self.df[nominal_columns]=LabelEncoder().fit_transform(self.df[nominal_columns].values)
        self.logger.info("nominal features Encoded successfully.")

    def encode_label(self, label: str) -> None:

        # Encode the target label using label encoding
        self.df[label]=LabelEncoder().fit_transform(self.df[label].values)
        self.logger.info("Encoded label successfully.")
        
    def retrieve_data(self) -> pd.DataFrame:

        # Return the processed dataset
        return self.df

    def shuffle(self) -> None:

        # Shuffle the dataset
        self.df = self.df.sample(frac = 1)

    def sample(self, red_factor: float) -> None:

        # Sample a fraction of the dataset
        self.df = self.df.sample(frac = red_factor)

    def scaling(self, column_list_to_be_scaled: List[str]) -> None:

        # Perform feature scaling on specified columns
        self.df[column_list_to_be_scaled]=StandardScaler().fit_transform(self.df[column_list_to_be_scaled])
        self.logger.info("Feature scaling completed successfully.")

    def splitting_data(self, percentage: float, label: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    
        # Split the dataset into train, validation, and test sets.

        # Parameters:
        #     percentage (float): The percentage of data to be allocated for training.
        #     label (str): The name of the target label column.

        # Returns:
        #     Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]: A tuple containing the following:
        #         - X_train: Training features DataFrame.
        #         - y_train: Training labels array.
        #         - X_val: Validation features DataFrame.
        #         - y_val: Validation labels array.
        #         - X_test: Test features DataFrame.
        #         - y_test: Test labels array.

        # Split the dataset into train, validation, and test sets
        y = self.df[label].values
        x = self.df[self.df.columns.difference([label])]
        
        # Split the data into train, remaining, and then further split remaining into validation and test
        X_train, X_remaining, y_train, y_remaining = train_test_split(x, y, test_size=percentage, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=1)

        return X_train, y_train, X_val, y_val, X_test, y_test 

    def plot_correlationMatrix(self) -> None:

        # Plot correlation matrix heatmap
        correlation_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True,)
        plt.title('Correlation Heatmap of DataFrame')
        plt.show()

    def plot_pairPlot (self) -> None:

        # Plot pairplot of different features
        sns.pairplot(self.df)
        plt.title('Pairplot of Different Features')
        plt.show()

    def plot_histogram_numerical(self) -> None:

        # Plot histograms for numerical columns
        numerical_columns = self.df._get_numeric_data().columns

        for column in self.df.numerical_columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    def plot_histogram_categorical(self) -> None:

        # Plot histograms for categorical columns
        categorical_columns = list(set(self.df.columns) - set(self.df._get_numeric_data().columns))
        
        for column in self.df.categorical_columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[column], kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    def plot_boxplot(self) -> None:

        # Plot boxplot of different features
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, orient='h')
        plt.title('Boxplot of Different Features')
        plt.show()

