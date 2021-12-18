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


def fbeta_macro(y_pred, y_test, beta=1, epsilon=1e-7) -> np.float32:
    precision = precision_macro(y_pred, y_test)
    recall = recall_macro(y_pred, y_test)
    return (1 + beta**2) * (precision*recall) / (beta**2 * precision + recall + epsilon)

def precision_macro(y_true, y_pred) -> np.float32:
    """ Computes macro-averaged precision score """
    assert y_true.shape == y_pred.shape, f"Dimension Mismatch!"

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    label = pd.Series(
        y_true,
        name = 'Actual'
    )
    
    pred = pd.Series(
        y_pred,
        name = 'Predicted'
    )

    cm = pd.crosstab(
        label,
        pred
    )

    cm = np.array(cm)

    return (cm.diagonal() / cm.sum(axis=0)).mean()

def recall_macro(y_true, y_pred) -> np.float32:
    """ Computes macro-averaged recall score """

    assert y_true.shape == y_pred.shape, f"Dimension Mismatch!"

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    label = pd.Series(
        y_true,
        name = 'Actual'
    )

    pred = pd.Series(
        y_pred,
        name = 'Predicted'
    )

    cm = pd.crosstab(
        label,
        pred
    )

    cm = np.array(cm)

    return (conf_mat.diagonal() / conf_mat.sum(axis=1)).mean()

def precision_score(y_true, y_pred, epsilon=1e-7, average='micro') -> np.float32:
    if average == 'micro':
        tp = (y_true * y_pred).sum()
        tn = ((1 - y_true) * (1 - y_pred)).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        return tp / (tp + fp + epsilon)
    
    elif average == 'macro':
        return precision_macro(
            y_true, 
            y_pred
        )
    else:
        ValueError(f"Unknown average: {average}")


def recall_score(y_true, y_pred,  epsilon=1e-7, average='micro') -> np.float32:
    """ Computes recall score """
    if average == 'micro':
        tp = (y_true * y_pred).sum()
        tn = ((1 - y_true) * (1 - y_pred)).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        return tp / (tp + fn + epsilon)
    
    elif average == 'macro':
        return recall_macro(
            y_true,
            y_pred
        )
    
    else:
        ValueError(f"Unknown average: {average}")


def negative_predictive_value(y_true, y_pred,  epsilon=1e-7) -> np.float32:
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return tn / (tn + fn + epsilon)

def false_positive_rate(y_true, y_pred,  epsilon=1e-7) -> np.float32:
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return fp / (fp + tn + epsilon)

def false_discovery_rate(y_true, y_pred, epsilon=1e-7) -> np.float32:
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return fp / (fp + tp + epsilon)

def fbeta_score(y_true, y_pred, beta=1, epsilon=1e-7) -> np.float32:
    precision = precision_score(
        y_true,
        y_pred, 
    )

    recall = recall_score(
        y_true,
        y_pred, 
    )

    return (1 + beta**2) * (precision*recall) / (beta**2 * precision + recall + epsilon)

def accuracy(preds: Iterable[list or np.ndarray], labels: Iterable[list or np.ndarray], scale:bool = True) -> np.float:
    """Given the labels and predictions, calculate accuracy score. """
    return np.mean(preds == labels) * 100 if scale else np.mean(preds == labels)

def confusion_matrix(preds: Iterable[list or np.ndarray], labels: Iterable[list or np.ndarray]) -> pd.DataFrame:
    """Given the labels and predictions, calculate confusion matrix. """
    
    label = pd.Series(
        labels,
        name = 'Actual'
    )
    pred = pd.Series(
        preds,
        name = 'Predicted'
    )
    
    return pd.crosstab(
        label,
        pred
    )
