#importing the necessary packages 
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import matplotlib.pyplot as plt 

#creating datapoints
X,Y = make_classification(n_samples=20 , n_features=2 , n_informative=1 , n_redundant=0 , n_classes=2 ,n_clusters_per_class=1 , class_sep=10 , hypercube=False ,random_state=10)
scale = MinMaxScaler(feature_range=(1,10))
X_scaled = scale.fit_transform(X)

#creating scatterplot
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[Y == 0, 0], X_scaled[Y == 0, 1],color='blue', label='Class 0')
plt.scatter(X_scaled[Y == 1, 0], X_scaled[Y == 1, 1], color='red', label='Class 1')
plt.title("Classification Scatter Plot",fontsize =16)
plt.xlabel("Feature1",fontsize=14)
plt.ylabel("Feature2",fontsize=14)
plt.show()