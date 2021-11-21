#Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
heart_database = pd.read_csv("heart.csv")

#Feature and Traget Selection
X = heart_database.drop(columns='target', axis=1)
Y = heart_database['target']

#Train Test Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state = 2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Model Training
# from sklearn import svm
# model = svm.SVC(kernel='linear')
# model.fit(X_train, Y_train)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)


#accuracy on training data
from sklearn.metrics import accuracy_score
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Train Data Accuracy is", train_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test ,X_test_prediction)

print("Test Data Accuracy is", test_data_accuracy)

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_prediction))


input_data = [58,0,2,120,340,0,1,172,0,0,2,0,2]
def heart_pred(input_data):

    #input data as numpy array
    input_data_as_np = np.asarray(input_data)

    #reshape the array as we predicting only one instance
    input_data_reshaped = input_data_as_np.reshape(1,-1)

    #standardized the input data
    std_data = sc.transform(input_data_reshaped)

    # print(std_data)

    prediction = model.predict(std_data)
    # print(prediction)
    return prediction

prediction = heart_pred(input_data)

if(prediction[0]==0):
    print('Not a Heart Patient')
else:
    print('Heart Patient')

# Dumping Model
import joblib
joblib.dump(model,r"heart_model.joblib")



