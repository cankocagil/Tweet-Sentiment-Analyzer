from __future__ import (
    print_function,
    division
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random
import pickle
import json
import time
import abc
import sys
import os 

from collections import OrderedDict

from typing import (
    Callable,
    Iterable,
    List,
    Union,
    Tuple,
)

from utils import (
    Classifier,
    Pipeline,
    json_print,
    timeit,
    random_seed
)

class FeatureEngineer:
    def __init__(self):
        pass

    def fit(self, X):
        return NotImplementedError()

    def predict(self, X):
        return NotImplementedError()

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class OneHotEncoder:
    def __init__(self):
        pass
    def fit(self, X):
        self.n_values = np.max(X) + 1
        return self

    def transform(self, X):
        return np.eye(self.n_values)[X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    @staticmethod
    def one_hot(X, num_classes):
        return np.squeeze(np.eye(num_classes)[X.reshape(-1)])

class MinMaxScaler(FeatureEngineer):
    def __init__(self):
        pass

    def fit(self, X):
        self.min_ = np.min(X, axis = 0)
        self.max_ = np.max(X, axis = 0)
        return self

    def transform(self, X):
        return (X.copy() - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, X):
        return (X.copy() * (self.max_ - self.min_)) + self.min_


class StandardScaler(FeatureEngineer):
    def __init__(self):
        pass
    def fit(self, X):
        self.mean_ = np.mean(X, axis = 0)
        self.std_ = np.std(X, axis = 0)
        return self

    def transform(self, X):
        return (X.copy() - self.mean_) / self.std_

    def inverse_transform(self, X):
        return (X.copy() * self.std_) + self.mean_
