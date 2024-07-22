# -*- coding: utf-8 -*-
#data processing
#importing libraries
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#importing data set
data_set=pd.read_csv('Data.csv')
#extracting the independent variable
x= data_set.iloc[:,:-1].values  
print(x)
 
#extracting the dependent variable
y= data_set.iloc[:,3].values 
print(y)

#handling missing data (replacing missing data with the mean value)
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
    #Fitting the imputer object tto the independent variable x.
imputer= imputer.fit(x[:,1:3])
    #replacing missing data with the calculated mean value
x[:, 1:3]=imputer.transform(x[:, 1:3])
print("Independent variables after handling missing data(x):\n")
print(x)

#encoding categorical data
#for country variable
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0]) 
print(x)

#encoding for dummy variables
print("\n")
onehot_encoder= OneHotEncoder()
x= onehot_encoder.fit_transform(x).toarray()
print(x)

#for the purchased variable:
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
print("\n")
print(y)

#splitting dataset into training set and test set
X_train, X_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=0)
#X_train: features for training data
#X_test: features for testing data
#y_train: Dependent variables for training data
#y_test: Independent variable for testing data
print("\n")
#printing the shapes of resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#feature scaling(final step of data processing)
#either by normalization or standardization
#let's use standardization
st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)  
X_test= st_x.transform(X_test)  
print(X_train)
print(X_test)
