

from Analyzer import Analyzer
from Regressor import ANN_keras_regressor
from Regressor import Random_forest_regressor




#######  regression  ########

diamonds2 = Analyzer(r"C:\Users\satin\ML_framework\ML_framework\ML_framework\diamonds.csv")
diamonds2.read_dataset()
print(diamonds2.retrieve_data())
print(diamonds2.describe())


################ pre_processing #############
diamonds2.drop_missing_data()
diamonds2.drop_columns("Unnamed: 0")
diamonds2.encode_features_categorical(["color","clarity","cut"])
print(diamonds2.retrieve_data())


########### Normalizing #######
diamonds2.scaling(diamonds2.retrieve_data().columns.difference(["price"]))
print(diamonds2.retrieve_data())

############# Splitting data #################
X_train2, y_train2, X_val2, y_val2, X_test2, y_test2  =  diamonds2.splitting_data(0.30,"price")

regressor1 = ANN_keras_regressor()

regressor1.add_layers(layer_units=[100, 50,1], activations=['relu','relu' ,'linear'])
regressor1.compile(loss='mean_squared_error', optimizer='adam')
regressor1.fit(X_train2,y_train2,epochs=250,verbose=1)
regressor1.predict(X_test2)
print("Mean-squared error, R2 score, Mean absolute error are :",regressor1.score(X_test2,y_test2))


#regressor2 = Random_forest_regressor()
#regressor2.hypertune_params(X_train2, y_train2, X_test2, y_test2)
#regressor2.fit(X_train2,y_train2)
#regressor2.predict(X_test2)
#print("accuracy score",regressor2.score(X_test2,y_test2))






