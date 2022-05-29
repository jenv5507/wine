import pandas as pd
import math
import numpy as np
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression


#  Import and read the csv file(s)
#file_path = "../Mod20_Group_Challenge/iris.csv"
wine_df=pd.read_csv("https://raw.githubusercontent.com/saraegregg/Mod20_Group_Challenge/main/data_cleaning/cleaned_wine_data.csv")
wine_df.head(10)

# Pull in top keywords
keywords = ['ripe', 'crisp', 'bright', 'dry', 'full', 'sweet', 'fresh', 'earthy', 'bitter', 'aftertaste']
for k in keywords:
    wine_df[k] = wine_df.description.apply(lambda x : 1 if x.find(k)>-1 else 0)
wine_df.head()

def getPoints(points):
    if(points <= 85):
        return 'bad'
    elif(points<=90 ):
        return 'ok'
    elif(points<=95):
        return 'good'
    elif(points<=100):
        return 'great'
    else:
        return 'If this gets hit, we did something wrong!'

wine_df['Points'] = wine_df['points'].apply(getPoints)


country_counts=wine_df.country.value_counts()
country_counts

# Determine which values to replace if counts are less than ..?
replace_country = list(country_counts[country_counts < 2000].index)

# Replace in dataframe
for coun in replace_country:
    wine_df.country= wine_df.country.replace(coun,"Other")
    
# Check to make sure binning was successful
wine_df.country.value_counts()

variety_counts=wine_df.variety.value_counts()
variety_counts

# Determine which values to replace if counts are less than ..?
replace_variety = list(variety_counts[variety_counts < 2000].index)

# Replace in dataframe
for var in replace_variety:
    wine_df.variety= wine_df.variety.replace(var,"Other")
    
# Check to make sure binning was successful
wine_df.variety.value_counts()

wine_df = wine_df.drop(columns=['description', 'province', 'title', 'winery', 'taster_name'], axis=1) 

# # hot encoding for country and taster name as they are limited categories. 
wine_df = pd.get_dummies(wine_df, columns=['country', 'variety'])

# Split our preprocessed data into our features and target arrays
y = wine_df["Points"].values
X = wine_df.drop(["Points"],1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

lr = LogisticRegression(multi_class='ovr',solver='lbfgs',max_iter=100)
model = lr.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))

import pickle
pickle.dump(lr, open("../model.p", "wb"))

