# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:05:48 2021

@author: SIMU
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics 

# import data
dataset = pd.read_csv('C:/Users/SIMU/Desktop/portfolio/salary prediction/hiring.csv')
dataset.head(n=12)

# get information about data
dataset.info()

# check null values
print(dataset.isnull().sum())

#fill experience null values with 0
dataset['experience'].fillna(0, inplace=True)

#fill test score null value with mean
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

#convert categorical value to numeric
def convert_to_numeric(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

dataset['experience'] = dataset['experience'].apply(lambda x : convert_to_numeric(x))

# check null values
print(dataset.isnull().sum())

#pairplot to see how different columns are related and how their distribution looks like
sns.pairplot(dataset[['test_score','interview_score']] , diag_kind="kde");

#Model building
#Create input feature
X = dataset.iloc[:, :3]
#target feature
y = dataset.iloc[:,-1]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# model training
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X_train, y_train)

# Visualising the Training set results
plt.scatter(X_train['experience'], y_train, color = 'red')
plt.scatter(X_train['experience'], regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_pred = regressor.predict(X_test)
b0 = regressor.intercept_
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

# Visualising the test set results
plt.scatter(X_test['experience'], y_test, color = 'red')
plt.scatter(X_test['experience'], regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Saving model to disk
pickle.dump(regressor, open("C:/Users/SIMU/Desktop/portfolio/salary prediction/model.pkl", "wb"))

# Loading model to compare the results
model = pickle.load(open("C:/Users/SIMU/Desktop/portfolio/salary prediction/model.pkl", "rb"))
print(model.predict([[10, 7, 6]]))

















