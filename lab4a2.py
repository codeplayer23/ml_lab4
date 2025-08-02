# importing the necessary packages
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error , root_mean_squared_error , r2_score , mean_absolute_percentage_error

# loading the datset
file_path = "/Users/niteshnirranjan/Downloads/Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

# extracting feature vectors 
X = df.iloc[:,3:7]
Y = df.iloc[:,8:9]

# splitting dataset into training and test set
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

# KNN regressor 
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train,Y_train)

# predicting the chg%
Y_pred = knn.predict(X_test)

# Mean Squared Error 
mse = mean_squared_error(Y_test,Y_pred)
print("Mean Square Error :",mse)

# Mean Absolute Percentage Error 
mape = mean_absolute_percentage_error(Y_test,Y_pred)
print("Mean Absolute Percentage Error :",mape)

# Root Mean Squared Error 
rmse = root_mean_squared_error(Y_test,Y_pred)
print("Root Mean Squared Error :",rmse)

# R2 score 
r2 = r2_score(Y_test,Y_pred)
print("R2 Score :",r2)