# -*- coding: utf-8 -*-
'''
Name:Xieqin You
ID: 2021423425
class: CS6364
HW5 Question 1
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from normalizeFeatures import normalizeFeatures

from gradientDescent import gd_for_linear_reg
from sgd import sgd_for_linear_reg
from sgd_with_momentum import sgd_with_momentum_for_linear_reg
from sgd_with_nesterov_momentum import sgd_with_nesterov_for_linear_reg
from adaGrad import adaGrad_for_linear_reg

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

sns.set(rc = {'figure.figsize': (12, 9)})
sns.distplot(boston['MEDV'], bins=30)

corr_matrix = boston.corr().round(2)
sns.heatmap(corr_matrix, annot = True)

X = boston.drop(['MEDV'], axis = 1)
y = boston['MEDV']

print("Normalizing features...")
X = normalizeFeatures(X)

# Add intercept term to X
X.insert(0, 'Ones', 1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
m = y_train.size

#--------------------1. Gradient Descent-----------------------------------------------
print("1. Running gradient descent ...")

alpha = 0.005
epochs = 1000
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = gd_for_linear_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)

rmse_train = np.sqrt(cost)

print('Using gradient descent, the trained theta is: ')
print(theta)
print('\n')

print('Using gradient descent, the model performance for training set: ')
print('RSME is {}'.format(rmse_train))
print('\n')

test_cost = np.sum(np.square(np.dot(X_test, theta) - y_test)) / (2 * m)
rmse_test = np.sqrt(test_cost)

print('Using gradient descent, the model performance for test set: ')
print('RSME is {}'.format(rmse_test))
print('\n')

with open('linear_reg_result.txt', 'w') as f:
    f.write('Using gradient descent, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('RSME for training set:' + str(rmse_train) + '\n')
    f.write('RSME for test set: '+ str(rmse_test) + '\n')
    f.write('\n\n')
    
#--------------------2. SGD (Stachastic Gradient Descent) -----------------------------------------------
print("2. Running Stachastic Gradient Descent(SGD) ...")

alpha = 0.005
epochs = 1000
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_for_linear_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)


rmse_train = np.sqrt(cost)

print('Using stachostic gradient descent (SGD), the trained theta is: ')
print(theta)
print('\n')

print('Using stachostic gradient descent (SGD), the model performance for training set: ')
print('RSME is {}'.format(rmse_train))
print('\n')

test_cost = np.sum(np.square(np.dot(X_test, theta) - y_test)) / (2 * m)
rmse_test = np.sqrt(test_cost)

print('Using stachostic gradient descent (SGD), the model performance for testing set: ')
print('RSME is {}'.format(rmse_test))
print('\n')

with open('linear_reg_result.txt', 'a') as f:
    f.write('Using stachostic gradient descent (SGD), the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('RSME for training set:' + str(rmse_train) + '\n')
    f.write('RSME for test set: '+ str(rmse_test) + '\n')
    f.write('\n\n')

#--------------------3. SGD (Stachastic Gradient Descent with Momentum) -----------------------------------------------
print("3. Running Stachastic Gradient Descent(SGD) with Momentum...")
alpha = 0.005
epochs = 500
velocity = np.zeros(X_train.shape[1])
decay_factor = 0.9
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_with_momentum_for_linear_reg(X_train.values, y_train.values, theta, alpha, velocity, decay_factor, epochs)

plt.plot(cost_history)

rmse_train = np.sqrt(cost)

print('Using SGD with momentum, the trained theta is: ')
print(theta)
print('\n')

print('Using SGD with momentum, the model performance for training set: ')
print('RSME is {}'.format(rmse_train))
print('\n')

test_cost = np.sum(np.square(np.dot(X_test, theta) - y_test)) / (2 * m)
rmse_test = np.sqrt(test_cost)

print('Using SGD with momentum, the model performance for testing set: ')
print('RSME is {}'.format(rmse_test))
print('\n')

with open('linear_reg_result.txt', 'a') as f:
    f.write('Using SGD with momentum, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('RSME for training set:' + str(rmse_train) + '\n')
    f.write('RSME for test set: '+ str(rmse_test) + '\n')
    f.write('\n\n')

#--------------------4. SGD (Stachastic Gradient Descent with Momentum) -----------------------------------------------
print("4. Running Stachastic Gradient Descent(SGD) with Nesterov Momentum...")
alpha = 0.005
epochs = 500
velocity = np.zeros(X_train.shape[1])
decay_factor = 0.9
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = sgd_with_nesterov_for_linear_reg(X_train.values, y_train.values, theta, alpha, velocity, decay_factor, epochs)

plt.plot(cost_history)

rmse_train = np.sqrt(cost)

print('Using SGD with Nesterov momentum, the trained theta is: ')
print(theta)
print('\n')

print('Using SGD with Nesterov momentum, the model performance for training set: ')
print('RSME is {}'.format(rmse_train))
print('\n')

test_cost = np.sum(np.square(np.dot(X_test, theta) - y_test)) / (2 * m)
rmse_test = np.sqrt(test_cost)

print('Using SGD with Nesterov momentum, the model performance for testing set: ')
print('RSME is {}'.format(rmse_test))
print('\n')

with open('linear_reg_result.txt', 'a') as f:
    f.write('Using SGD with Nesterov momentum, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('RSME for training set:' + str(rmse_train) + '\n')
    f.write('RSME for test set: '+ str(rmse_test) + '\n')
    f.write('\n\n')    
    
#--------------------5. AdaGrad -----------------------------------------------------------
print("5. AdaGrad Algorithm")
alpha = 0.3
epochs = 4000
theta = np.zeros(X_train.shape[1])
theta, cost, cost_history = adaGrad_for_linear_reg(X_train.values, y_train.values, theta, alpha, epochs)

plt.plot(cost_history)

rmse_train = np.sqrt(cost_history[len(cost_history) - 1])

print('Using AdaGrad, the trained theta is: ')
print(theta)
print('\n')


print('Using AdaGrad, the model performance for training set: ')
print('RSME is {}'.format(rmse_train))
print('\n')

test_cost = np.sum(np.square(np.dot(X_test, theta) - y_test)) / (2 * m)
rmse_test = np.sqrt(test_cost)

print('Using AdaGrad, the model performance for testing set: ')
print('RSME is {}'.format(rmse_test))
print('\n')

with open('linear_reg_result.txt', 'a') as f:
    f.write('Using AdaGrad, the trained theta is: \n')
    f.write(str(theta))
    f.write('\n')
    f.write('RSME for training set:' + str(rmse_train) + '\n')
    f.write('RSME for test set: '+ str(rmse_test) + '\n')
    f.write('\n\n')       
    
    
    
    