#importing the necessary packages 
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#creating datapoints
X,Y = make_classification(n_samples=1000 , n_features=2 , n_informative=1 , n_redundant=0 , n_classes=2 ,n_clusters_per_class=1 , class_sep=10 , hypercube=False ,random_state=10)
scale = MinMaxScaler(feature_range=(1,10))
X_scaled = scale.fit_transform(X)
X_scaled = (X_scaled * 10).round() / 10

# splitting dataset into training and test set
X_train , X_test , Y_train , Y_test = train_test_split(X_scaled,Y,test_size=0.2)

# KNN regressor 
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)

#creating scatterplot
plt.figure(figsize=(10,6))
plt.scatter(X_test[Y_test == 0][:, 0], X_test[Y_test == 0][:, 1], color='red', label='Actual Class 0', alpha=0.6)
plt.scatter(X_test[Y_test == 1][:, 0],  X_test[Y_test == 1][:, 1], color='blue',  label='Actual Class 1',  alpha=0.6)
plt.scatter(X_test[Y_pred != Y_test][:, 0], X_test[Y_pred != Y_test][:, 1],  facecolors='none', edgecolors='black', linewidths=1.5, label='Misclassified')
plt.title("Actual vs predicted",fontsize =16)
plt.xlabel("Feature1",fontsize=14)
plt.ylabel("Feature2",fontsize=14)
plt.show()
