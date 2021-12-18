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

from preprocessing import (
    StandardScaler,
    MinMaxScaler,
    FeatureEngineer
)

class PCA(FeatureEngineer):
    def __init__(self, n_components:int = 3):
        self.n_components_ = n_components

    def fit(
        self,
        X,
    ):
        
        self.n_obs, self.n_features = X.shape
    
        self.scaler = StandardScaler()

        self.X_ = self.scaler.fit_transform(X)

        self.cov_mat_ = np.cov(
            self.X_,
            rowvar = False
        )

        eigen_values, eigen_vectors = np.linalg.eigh(self.cov_mat_)

        sorted_index = np.argsort(eigen_values)[::-1]

        self.sorted_eigenvalue_ = eigen_values[sorted_index]
        self.sorted_eigenvectors_ = eigen_vectors[:, sorted_index]

        self.principal_components_ = self.sorted_eigenvectors_[:, :self.n_components_]

        self.explained_variance_ratio_ = self.sorted_eigenvalue_ / np.sum(self.sorted_eigenvalue_, axis=0)
        
        self.cum_explained_variance_ratio_ = np.cumsum(
            self.explained_variance_ratio_ ,
            axis=0
        )

        return self

    def transform(self, X):
        norm_X = (X - self.scaler.mean_) / self.scaler.std_
        return norm_X  @ self.principal_components_

    def inverse_transform(self, X):
        construct_ = X @ self.principal_components_.T
        return (construct_ * self.scaler.std_) + self.scaler.mean_


    def get_eigen_faces(self):
        image_dim = np.int(
            np.sqrt(self.n_features)
        )

        eigen_faces = np.reshape(
            self.principal_components_.T, 
            [-1, image_dim, image_dim]
        )

        return eigen_faces


    def __str__(self):
        return f"Principal Component Analysis (PCA) with {self.n_components_} components"

    def __repr__(self):
        return f"Principal Component Analysis (PCA) with {self.n_components_} components"
