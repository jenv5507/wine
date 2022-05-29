from telnetlib import XDISPLOC
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#Read in our data file and set it up. 
df = pd.read_csv("diabetes.csv")
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

#Set up / train the model.  was all taken from the exercise
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, y_train)

y_true = y_test
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

print(cm_df)

import pickle
pickle.dump(classifier, open("../model.p", "wb"))

featureList = [] 

for col in X.columns: 
    feature = {
        "Name" : col, 
        "Label" : col, 
        "Value" : 0, 
        "Min" : X[col].min(),
        "Max" : X[col].max()
    }

    featureList.append(feature)

pd.DataFrame(featureList).to_csv("../features.csv", index=False)