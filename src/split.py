from __future__ import print_function, division

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
import os 

from typing import (
    Callable,
    Iterable,
    List,
    Union,
    Tuple,
)
from collections import OrderedDict


def train_test_split(X:pd.DataFrame, split_size:list = [0.7, 0.1, 0.2], random_state:int = 42):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X.copy())

    split_size = np.array(split_size)

    assert np.isclose(split_size.sum(), 1.0), f"Split ratios should sum to 1"

    cum_splits = split_size.cumsum()
    len_X = len(X)

    indices = np.ceil(cum_splits[:2] * len_X)

    train, val, test = np.split(
        X.sample(frac = 1, random_state = random_state),
        indices.astype(np.int)
    )
    
    return train, val, test