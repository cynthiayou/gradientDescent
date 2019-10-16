# -*- coding: utf-8 -*-
import numpy as np
import random
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sgd_with_nesterov_for_linear_reg(X, y, theta, alpha, velocity, decay_factor, epochs):
    m = y.size
    cost_history = []
    best_theta = 0
    cost_threshold = float('inf')
    for i in range(epochs):
        idx = random.randint(1, m) - 1
        sample_X = X[idx]
        sample_y = y[idx]
        h = np.dot(sample_X, theta + decay_factor * velocity)  
        velocity = decay_factor * velocity - alpha * (np.dot(sample_X.T, (h - sample_y)))
        theta = theta + velocity
        cost = np.sum(np.square(np.dot(X, theta) - y)) / (2 * m)
        if (cost < cost_threshold):
            best_theta = theta
            cost_threshold = cost
        cost_history.append(cost)
    
    return best_theta, cost_threshold, cost_history


def sgd_with_nesterov_for_logistic_reg(X, y, theta, alpha, velocity, decay_factor, epochs):
    m = y.size
    cost_history = []
    best_theta = 0
    cost_threshold = float('inf')
    for i in range(epochs):
        idx = random.randint(1, m) - 1
        sample_X = X[idx]
        sample_y = y[idx]
        h = np.vectorize(sigmoid)(np.dot(sample_X, theta + decay_factor * velocity))  
        velocity = decay_factor * velocity - alpha * (np.dot(sample_X.T, (h - sample_y)))
        theta = theta + velocity
        new_h = np.vectorize(sigmoid)(np.dot(X, theta))
        cost = np.sum(y * np.log(new_h) + (1 - y)*np.log(1 - new_h)) / (-2 * m)
        if (cost < cost_threshold):
            best_theta = theta
            cost_threshold = cost
        cost_history.append(cost)
              
    return best_theta, cost_threshold, cost_history
