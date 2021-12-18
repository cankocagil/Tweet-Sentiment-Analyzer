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
    MinMaxScaler
)

from loss import MeanSquaredError



class LinearRegression(object):
    
    """
        Ridge regression is a method of estimating the coefficients of multiple-regression models in
        scenarios where independent variables are highly correlated. 

    """
    def __init__(self, Lambda:float=0):
        """
            Constructer method for initilization of ridge regression model.
            
            
                Arguments:
                    - Lambda (float): is the parameter which balances the amount 
                     of emphasis given to minimizing RSS vs minimizing sum of square of coefficients

        
        """

        self.Lambda = Lambda      
     
    def fit(self, X:np.ndarray, y:np.ndarray, concat_ones:bool = False) -> None:
        """
            
            Given the pair of X,y, fit the data, i.e., find parameter W such that sum of square error
            is minimized. 
            

                Arguments:
                    - X (np.ndarray) : Regressor data 
                    - X (np.ndarray) : Ground truths for regressors

                Returns:
                    - None
        
        """
        
        I = np.eye(X.shape[1])
        
        self.W = np.linalg.inv(
            X.T.dot(X) + self.Lambda * I
        ).dot(X.T).dot(y)

        return self

    def predict(self,X:np.ndarray, concat_ones:bool = False) -> np.ndarray:
        """
            Given the test data X, we predict the target variable.
            
                Arguments:
                    - X (np.ndarray) : The independant variable (regressor)

                Returns:
                    - Y_hat (np.ndarray) : Estimated value of y

        """
        return X.dot(self.W)
    

    def parameters(self) -> None:
        """
            Returns the estimated parameter W of the Ridge Regression
        
        """
        return self.W

    def eval_r2(self,y_true:np.ndarray, y_pred:np.ndarray) -> np.float:
        """
            Given the true dependant variable and estimated variable, computes proportion of
            explained variance R^2 by square the Pearson correlation between true dependant
            variable and estimated variabl
            
                Arguments:
                    - y_true (np.ndarray) : true dependant variable
                    - y_pred (np.ndarray) : estimated variable
                    
                Returns:
                    - r_squared (np.float) : Proportion of explained variance
        
        """

        _pearson = np.corrcoef(y_true, y_pred)
        pearson = _pearson[1][0]
        r_squared = np.square(pearson)
        return r_squared

    @staticmethod
    def R2(y_true:np.ndarray,y_pred:np.ndarray) -> np.float:
        r_squared = (1 - (sum((y_true - (y_pred))**2) / ((len(y_true) - 1) * np.var(y_true.T, ddof=1)))) * 100
        return r_squared


    def __str__(self):
        model = LinearRegression().__class__.__name__
        model += f" with parameter \n"
        model += f"{self.Lambda}"
        return model


    def __repr__(self):
        model = LinearRegression().__class__.__name__
        model += f" with parameter \n"
        model += f"{self.Lambda}"
        return model