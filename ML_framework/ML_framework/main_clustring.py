

import ast
from Analyzer import Analyzer
from Clustering import DBscan
from Clustering import Kmeans


########  clustring  ########

analysed1 = Analyzer(r"C:\Users\satin\ML_framework\ML_framework\ML_framework\diamonds.csv")
analysed1.read_dataset()
print(analysed1.retrieve_data())
print(analysed1.describe())
################# pre_processing #############
analysed1.drop_missing_data()
analysed1.drop_columns("Unnamed: 0")
analysed1.encode_features_ordinal(["color","clarity","cut"])
analysed1.scaling(analysed1.retrieve_data().columns)

print(analysed1.retrieve_data())

clusters_model = Kmeans()
clusters_model.find_k(analysed1.retrieve_data())
clusters_model.fit_predict(input("Enter the best value of k: "))

print("silhouette_score",clusters_model.calculate_silhouette_score())

#clusters_model1 = DBscan()
#clusters_model1.hypertune_params(analysed1.retrieve_data())
#clusters_model1.fit_predict(input("give the best params"))
#clusters_model1.calculate_silhouette_score()
#clusters_model1.visualize()







