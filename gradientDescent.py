# -*- coding: utf-8 -*-
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def gd_for_linear_reg(X, y, theta, alpha, epochs):
    m = y.size
    cost_history = []
    best_theta = 0
    cost_threshold = float('inf')
    for i in range(epochs):
        h = np.dot(X, theta)      
        theta = theta - (alpha / m) * (np.dot(X.T, (h - y)))
        cost = np.sum(np.square(np.dot(X, theta) - y)) / (2 * m)
        if (cost < cost_threshold):
            best_theta = theta
            cost_threshold = cost
        cost_history.append(cost)
    
    return best_theta, cost_threshold, cost_history

def gd_for_logistic_reg(X, y, theta, alpha, epochs):
    m = y.size
    cost_history = []
    best_theta = 0
    cost_threshold = float('inf')
    for i in range(epochs):
        h = np.vectorize(sigmoid)(np.dot(X, theta))    
        theta = theta - (alpha / m) * (np.dot(X.T, (h - y)))
        new_h = np.vectorize(sigmoid)(np.dot(X, theta))
        cost = np.sum(y * np.log(new_h) + (1 - y)*np.log(1 - new_h)) / (-2 * m)
        if (cost < cost_threshold):
            best_theta = theta
            cost_threshold = cost
        cost_history.append(cost)
    
    return best_theta, cost_threshold, cost_history

