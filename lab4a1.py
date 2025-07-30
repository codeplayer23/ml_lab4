#importing the required packages 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score ,accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#loading the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#extracting feature vectors 
X = df.iloc[:,:196]
Y = df["LABEL"]

#splitting into training and test 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#KNN Classifier 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_train_pred = knn.predict(X_train)
Y_test_pred = knn.predict(X_test)

#confusion matrix 
cm = confusion_matrix(Y_test,Y_test_pred)
print(cm)

#precision metrics 
train_precision = precision_score(Y_train,Y_train_pred,average="weighted")
test_precision = precision_score(Y_test,Y_test_pred,average="weighted")
train_recall = recall_score(Y_train,Y_train_pred,average="weighted")
test_recall = recall_score(Y_test,Y_test_pred,average="weighted")
train_f1 = f1_score(Y_train,Y_train_pred,average="weighted")
test_f1 = f1_score(Y_test,Y_test_pred,average="weighted")

#accuracy score 
train_accuracy = accuracy_score(Y_train,Y_train_pred)
test_accuracy = accuracy_score(Y_test,Y_test_pred)

print("Training precision :",train_precision)
print("Testing precision :",test_precision)
print("Training recall :",train_recall)
print("Testing recall :",test_recall)
print("Training f1 score :",train_f1)
print("Testing f1 score :",test_f1)
print("Training Accuracy :",train_accuracy)
print("Testing Accuracy :",test_accuracy)

if(train_accuracy>test_accuracy):
    print("overfitting")
if (train_accuracy==test_accuracy):
    print("underfitting")
if(train_accuracy<test_accuracy):
    print("overfitting")