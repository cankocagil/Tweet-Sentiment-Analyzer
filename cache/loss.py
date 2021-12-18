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

from preprocessing import StandardScaler, MinMaxScaler

class MeanSquaredError:
    def __init__(self):
        self.is_computed_ = False

    def compute(
        self,
        y_true: np.ndarray, 
        y_pred: np.ndarray
    
    ):
        """
            Given the grounth truth matrix and prediction, computes element wise squared error.
            
                
            Arguments:
                - y_true  (np.ndarray) : grounth truth
                - y_pred  (np.ndarray) : prediction
                
            Returns:
                square_error (np.ndarray) : Point-wise MSE loss
        
        """
        assert y_true.shape == y_pred.shape, f'Mismatch Dimension!, {y_true.shape} does not match with {y_pred.shape}'
        self.mse = (y_true - y_pred) ** 2
        self.is_computed_ = True

        return self.mse
    
    def stats(self):
        assert self.is_computed_, "MSE is not computed"
        
        per_obs_mean = np.mean(self.mse, axis = 1)
        
        return {
            'mean': per_obs_mean.mean(), 
            'std': per_obs_mean.std()
        }