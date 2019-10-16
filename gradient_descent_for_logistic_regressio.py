# -*- coding: utf-8 -*-
'''
Name:Xieqin You
ID: 2021423425
class: CS6364
HW5 Question 2
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, accuracy_score

from normalizeFeatures import normalizeFeatures

from gradientDescent import gd_for_logistic_reg
from sgd import sgd_for_logistic_reg
from sgd_with_momentum import sgd_with_momentum_for_logistic_reg
from sgd_with_nesterov_momentum import sgd_with_nesterov_for_logistic_reg
from adaGrad import adaGrad_for_logistic_reg
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Load Data
train = pd.read_csv('titanic_train.csv')

# Data Proprocessing
# Check for null values
sns.set(rc={'figure.figsize': (12,8)})
sns.heatmap(train.isnull())
train.isnull().sum()
'''
# EDA
# Explore the correlation among variables
corr_matrix = train.corr()
sns.heatmap(corr_matrix, vmin = -0.7, vmax = 0.7, annot = True)
# Check the distibution of the target variable
sns.countplot(train['Survived'], palette='RdBu_r')
sns.countplot(train['Survived'], hue=train['Sex'], palette='RdBu_r')
#sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
'''
#Check if mean age differs by Pclass
sns.boxplot(x='Pclass', y='Age',data=train, palette='winter')

# Get the mean age by Pclass, and then fill the null value in Age column with this mean age
meanAgeByPclass = train[train['Age'].notnull()][['Age','Pclass']].groupby('Pclass').mean()
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        return meanAgeByPclass.loc[pclass]
    else:
        return age
 
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)

#Check if we have filled in all the null values in Age column
sns.heatmap(train.isnull())

#Drop the Cabin column as over 70% are null values
train.drop('Cabin', axis=1, inplace=True)

# Change Age from type Object to type int
train['Age'] = train['Age'].astype('int64')

# Convert categorical features
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)

# Drop the original categorical features
train.drop(['Name','Ticket', 'Sex', 'Embarked'], axis=1, inplace=True)

# Concatenate the dummy variables with train dataset
train = pd.concat([train, sex, embarked], axis=1)
train.head()

X = train.drop('Survived', axis=1)
y = train['Survived']

print("Normalizing features...")
X = normalizeFeatures(X)

# Add intercept term to X
X.insert(0, 'Ones', 1)

# Split the train dataset  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

m = y_train.size

#--------------------1. Gradient Descent-----------------------------------------------
print("1. Running gradient descent ...")

alpha = 0.01
epochs = 2000
theta = np.zeros(X_train.shape[1])

theta, cost, cost_history = gd_for_logistic_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)

print('Using gradient descent, the trained theta is: ')
print(theta)
print('\n')

y_train_pred = np.vectorize(sigmoid)(np.dot(X_train, theta))
y_train_pred[y_train_pred >= 0.5] = 1
y_train_pred[y_train_pred < 0.5] = 0
print('Using gradient descent, the model performance for training set: ')
report_train = classification_report(y_train, y_train_pred)
print(report_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: ", accuracy_train)
print('\n')

y_test_pred = np.vectorize(sigmoid)(np.dot(X_test, theta))
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
print('Using gradient descent, the model performance for test set: ')
report_test = classification_report(y_test, y_test_pred)
print(report_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: ", accuracy_test)
print('\n')

with open('logistic_reg_result.txt', 'w') as f:
    f.write('Using gradient descent, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('For training data: \n')
    f.write(report_train)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n')
    f.write('For test data: \n')
    f.write(report_test)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n\n')
    
#--------------------2. SGD (Stachastic Gradient Descent) -----------------------------------------------
print("2. Running Stachastic Gradient Descent(SGD) ...")

alpha = 0.01
epochs = 2000
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_for_logistic_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)

print('Using stachostic gradient descent (SGD), the trained theta is: ')
print(theta)
print('\n')

y_train_pred = np.vectorize(sigmoid)(np.dot(X_train, theta))
y_train_pred[y_train_pred >= 0.5] = 1
y_train_pred[y_train_pred < 0.5] = 0
print('Using stachostic gradient descent (SGD), the model performance for training set: ')
report_train = classification_report(y_train, y_train_pred)
print(report_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: ", accuracy_train)
print('\n')

y_test_pred = np.vectorize(sigmoid)(np.dot(X_test, theta))
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
print('Using stachostic gradient descent (SGD), the model performance for test set: ')
report_test = classification_report(y_test, y_test_pred)
print(report_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: ", accuracy_test)
print('\n')

with open('logistic_reg_result.txt', 'a') as f:
    f.write('Using stachostic gradient descent (SGD), the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('For training data: \n')
    f.write(report_train)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n')
    f.write('For test data: \n')
    f.write(report_test)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n\n')


#--------------------3. SGD (Stachastic Gradient Descent with Momentum) -----------------------------------------------
print("3. Running Stachastic Gradient Descent(SGD) with Momentum...")
alpha = 0.01
epochs = 2000
velocity = np.zeros(X_train.shape[1])
decay_factor = 0.5
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_with_momentum_for_logistic_reg(X_train.values, y_train.values, theta, alpha, velocity, decay_factor, epochs)

plt.plot(cost_history)

print('Using SGD with momentum, the trained theta is: ')
print(theta)
print('\n')

y_train_pred = np.vectorize(sigmoid)(np.dot(X_train, theta))
y_train_pred[y_train_pred >= 0.5] = 1
y_train_pred[y_train_pred < 0.5] = 0
print('Using SGD with momentum, the model performance for training set: ')
report_train = classification_report(y_train, y_train_pred)
print(report_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: ", accuracy_train)
print('\n')

y_test_pred = np.vectorize(sigmoid)(np.dot(X_test, theta))
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
print('Using SGD with momentum, the model performance for test set: ')
report_test = classification_report(y_test, y_test_pred)
print(report_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: ", accuracy_test)
print('\n')

with open('logistic_reg_result.txt', 'a') as f:
    f.write('Using SGD with momentum, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('For training data: \n')
    f.write(report_train)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n')
    f.write('For test data: \n')
    f.write(report_test)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n\n')
    
 
#--------------------4. SGD (Stachastic Gradient Descent with Momentum) -----------------------------------------------
print("4. Running Stachastic Gradient Descent(SGD) with Nesterov Momentum...")
alpha = 0.01
epochs = 2000
velocity = np.zeros(X_train.shape[1])
decay_factor = 0.5
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_with_nesterov_for_logistic_reg(X_train.values, y_train.values, theta, alpha, velocity, decay_factor, epochs)

plt.plot(cost_history)

print('Using SGD with Nesterov momentum, the trained theta is: ')
print(theta)
print('\n')

y_train_pred = np.vectorize(sigmoid)(np.dot(X_train, theta))
y_train_pred[y_train_pred >= 0.5] = 1
y_train_pred[y_train_pred < 0.5] = 0
print('Using SGD with Nesterov momentum, the model performance for training set: ')
report_train = classification_report(y_train, y_train_pred)
print(report_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: ", accuracy_train)
print('\n')

y_test_pred = np.vectorize(sigmoid)(np.dot(X_test, theta))
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
print('Using SGD with Nesterov momentum, the model performance for test set: ')
report_test = classification_report(y_test, y_test_pred)
print(report_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: ", accuracy_test)
print('\n')

with open('logistic_reg_result.txt', 'a') as f:
    f.write('Using SGD with Nesterov momentum, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('For training data: \n')
    f.write(report_train)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n')
    f.write('For test data: \n')
    f.write(report_test)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n\n')
    
     
#--------------------5. AdaGrad -----------------------------------------------------------    
print("5. AdaGrad Algorithm")
alpha = 0.1
epochs = 2000
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = adaGrad_for_logistic_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)

print('Using AdaGrad, the trained theta is: ')
print(theta)
print('\n')

y_train_pred = np.vectorize(sigmoid)(np.dot(X_train, theta))
y_train_pred[y_train_pred >= 0.5] = 1
y_train_pred[y_train_pred < 0.5] = 0
print('Using AdaGrad, the model performance for training set: ')
report_train = classification_report(y_train, y_train_pred)
print(report_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: ", accuracy_train)
print('\n')

y_test_pred = np.vectorize(sigmoid)(np.dot(X_test, theta))
y_test_pred[y_test_pred >= 0.5] = 1
y_test_pred[y_test_pred < 0.5] = 0
print('Using AdaGrad, the model performance for test set: ')
report_test = classification_report(y_test, y_test_pred)
print(report_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: ", accuracy_test)
print('\n')

with open('logistic_reg_result.txt', 'a') as f:
    f.write('Using AdaGrad, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('For training data: \n')
    f.write(report_train)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n')
    f.write('For test data: \n')
    f.write(report_test)
    f.write("Accuracy: " + str(accuracy_train))
    f.write('\n\n')

