#Source: https://curiousily.com/posts/build-a-decision-tree-from-scratch-in-python/
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest
import math
from sklearn import metrics  

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(h, y):
  return sqrt(mean_squared_error(h, y))

df_train = pd.read_csv('house_prices_train.csv')
df_test = pd.read_csv('house_prices_test.csv')


X = df_train[['OverallQual', 'GrLivArea', 'GarageCars']]
y = df_train['SalePrice']

#print(X, ' X')

class DecisionTreeRegressor:
    def fit(self, X, y, min_leaf = 5):
        self.dtree = Node(X, y, np.array(np.arange(len(y))), min_leaf)
        return self
    def predict(self, X): 
        return self.dtree.predict(X.values)

class Node:
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
    def find_varsplit(self):
        for c in range(self.col_count): self.find_better_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.min_leaf)
        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.min_leaf)
    def find_better_split(self, var_idx):
        x = self.x.values[self.idxs, var_idx]
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue
            curr_score = self.find_score(lhs, rhs)
            if curr_score < self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]
    def find_score(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]
    @property
    def is_leaf(self): return self.score == float('inf')
    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)


regressor = DecisionTreeRegressor().fit(X, y)
preds = regressor.predict(X)

metrics.r2_score(y, preds)
rmse(preds, y)


r2 = metrics.r2_score(y, preds)
rs = rmse(preds, y)

print(r2,rs)


