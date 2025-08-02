# importing the necessary packages 
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# loading the dataset
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

# extracting the feature vectors 
X = df.iloc[:,0:2]
Y = df["LABEL"]
print(X)

#splitting data into training and test set 
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#training KNN classifier
n = KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,Y_train)
Y_train_pred = n.predict(X_train)
Y_test_pred = n.predict(X_test)

#scatter plot
X_test_np = X_test.to_numpy()
Y_test_np = Y_test.to_numpy()
Y_pred_np = Y_test_pred
plt.figure(figsize=(10, 6))

plt.scatter(X_test_np[Y_test_np == 0][:, 0], X_test_np[Y_test_np == 0][:, 1], color='red',label='Actual Class 0',alpha=0.6)
plt.scatter( X_test_np[Y_test_np == 1][:, 0],X_test_np[Y_test_np == 1][:, 1],color='blue',label='Actual Class 1',alpha=0.6)
misclassified = Y_test_np != Y_pred_np
plt.scatter(X_test_np[misclassified][:, 0],X_test_np[misclassified][:, 1],facecolors='none',edgecolors='black',linewidths=1.5,label='Misclassified')
plt.title("KNN Classification Results on Test Set")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()