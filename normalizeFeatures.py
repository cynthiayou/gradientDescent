# -*- coding: utf-8 -*-
import numpy as np

def normalizeFeatures(X):
    mu = np.mean(X)
    sigma = np.std(X)
    
    X_norm = (X - mu) / sigma
    
    return X_norm