import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import optuna

class Clustering:
    def __init__(self) -> None:
        pass

    def calculate_silhouette_score(self) -> float:
        return silhouette_score(self.df,self.clusters)

    def visualize(self) -> None:

        for n in range(1,5):
            pca = PCA(n_components=n)
            df_pca = pca.fit_transform(self.df)

            print(pca.explained_variance_ratio_)
            print(sum(pca.explained_variance_ratio_))

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=self.clusters, palette='viridis', legend='full')
            plt.title('Cluster Visualization using PCA (DBSCAN)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(title='Clusters')
            plt.grid(True)
            plt.show()

class Kmeans(Clustering):
    def __init__(self) -> None:
        super().__init__()
    

    def find_k(self, df: pd.DataFrame) -> None:
        self.df = df
        # Initialize a list to hold inertia values for different k
        inertia_values = []

        # Try different values of k
        for k in range(1,50):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.df)
            inertia_values.append(kmeans.inertia_)
            print(f"For k={k}, inertia: {kmeans.inertia_}")

        # Visualize the elbow curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(inertia_values)), inertia_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Curve for Optimal k')
        plt.xticks(range(len(inertia_values)))
        plt.grid(True)
        plt.show()

        print(f"elbow curve is: {inertia_values}")
        

    def fit_predict(self, k) -> None:
        kmeans2 = KMeans(n_clusters=k)
        kmeans2.fit(self.df)
        self.clusters = kmeans2.labels_


class DBscan(Clustering):
    def __init__(self) -> None:
        super().__init__()
        

    def hypertune_params(self, df: pd.DataFrame) -> None:
        self.df = df
        def objective(trial):
            #eps = trial.suggest_uniform('eps', 0.1, 1.0)
            #min_samples = trial.suggest_int('min_samples', 2, 10)

            #dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan = DBSCAN()
            clusters = dbscan.fit_predict(df)

            return silhouette_score(df, clusters)

        # Create Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        # Get the best parameters
        best_params = study.best_params
        best_score = study.best_value

        print("Best Parameters:", best_params)
        print("Best Silhouette Score:", best_score)

    
    def fit_predict(self,**params) -> None:
        self.clustring_type = DBSCAN(**params)
        self.clusters = self.clustring_type.fit_predict(self.df)

class Agglomerative_hierarchal(Clustering):
    def __init__(self) -> None:
        super().__init__()

    def hypertune_params(self, df: pd.DataFrame) -> None:
        self.df = df
        def objective(trial):
            # Define hyperparameters to tune
            n_clusters = trial.suggest_int('n_clusters', 2, 10)

            # Initialize Agglomerative Hierarchical Clustering with hyperparameters
            clustering = AgglomerativeClustering(n_clusters=n_clusters)

            # Fit clustering algorithm to data and get labels
            labels = clustering.fit_predict(df)
            score = silhouette_score(df, labels)

            return score

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # Retrieve the best hyperparameters
        best_params = study.best_params
        best_score = study.best_value

        print("Best Parameters:", best_params)
        print("Best Silhouette Score:", best_score)

    def fit_predict(self,**params) -> None:
        self.clustring_type = AgglomerativeClustering(**params)
        self.clusters = self.clustring_type.fit_predict(self.df)


class Mean_shift(Clustering):
    def __init__(self) -> None:
        super().__init__()


    def hypertune_params(self, df: pd.DataFrame) -> None:
        self.df = df
        def objective(trial):
            # Define hyperparameters to tune
            bandwidth = trial.suggest_float('bandwidth', 2, 10)

            # Initialize Mean Shift Clustering with hyperparameters
            clustering = MeanShift(bandwidth=bandwidth)

            # Fit clustering algorithm to data and get labels
            labels = clustering.fit_predict(df)

            # Calculate a score to be maximized or minimized (e.g., silhouette score)
            # You can use a different metric based on your requirements
            score = silhouette_score(df, labels)

            return score

        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # Retrieve the best hyperparameters
        best_params = study.best_params
        best_score = study.best_value

        print("Best Parameters:", best_params)
        print("Best Silhouette Score:", best_score)

    def fit_predict(self,**params) -> None:
        self.clustring_type = MeanShift(**params)
        self.clusters = self.clustring_type.fit_predict(self.df)








