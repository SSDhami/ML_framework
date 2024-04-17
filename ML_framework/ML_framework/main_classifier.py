


from Analyzer import Analyzer
from Classifier import KNN_classifier
from Classifier import ANN_sklearn
from Classifier import ANN_keras




#############reading dataset##################
diamonds_df = Analyzer(r"C:\Users\satin\ML_framework\ML_framework\ML_framework\diamonds.csv")
diamonds_df.read_dataset()
print(diamonds_df.retrieve_data())
print(diamonds_df.describe())


################ pre_processing #############
diamonds_df.drop_missing_data()
diamonds_df.drop_columns("Unnamed: 0")
diamonds_df.encode_label("cut")
diamonds_df.encode_features_categorical(["color","clarity"])
print(diamonds_df.retrieve_data())


########################## Plotting ####################
diamonds_df.plot_correlationMatrix()
#diamonds_df.plot_pairPlot()
#diamonds_df.plot_histogram_numerical()
#diamonds_df.plot_histogram_categorical()
#diamonds_df.plot_boxplot()

########### Normalizing #######
diamonds_df.scaling(diamonds_df.retrieve_data().columns.difference(["cut"]))
print(diamonds_df.retrieve_data())

############# Splitting data #################
X_train, y_train, X_val, y_val, X_test, y_test = diamonds_df.splitting_data(0.20,"cut")


#######  Classification  ########
classifier = ANN_sklearn(hidden_layer_sizes=(42) ,max_iter=1000)
classifier.fit(X_train,y_train)
classifier.predict(X_test)
print("Accuracy score, Precision score, Recall score, F1-score are:",classifier.score(X_test,y_test))

#classifier2 = KNN_classifier()
#classifier2.hypertune_params(x_train, x_test, y_train, y_test)
#classifier2.fit(x_train,y_train)
#classifier2.predict(x_test)
#print("accuracy score",classifier2.score(x_test,y_test))


#classifier3 = ANN_keras()

#classifier3.add_layers(layer_units=[100, 50,5], activations=['relu','relu' ,'softmax'])
#classifier3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#classifier3.fit(x_train,y_train,epochs=5000,verbose=1)
#classifier3.predict(x_test)
#print(classifier3.score(y_test))


