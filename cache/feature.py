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

def mutual_information(
    x1:np.ndarray,
    x2:np.ndarray
) -> np.float:

    jh = np.histogram2d(
        x1,
        x2,
        bins = (
            256, 256
        )
    )[0]

    jh = jh + 1e-15

    sh = np.sum(jh)
    jh = jh / sh

    y1 = np.sum(
        jh,
        axis=0
    ).reshape(
        ( -1, jh.shape[0])
    )

    y2 = np.sum(
        jh,
        axis=1
    ).reshape(
        (jh.shape[1], -1)
    )


    return  ( 
        np.sum(jh * np.log(jh)) - np.sum(y1 * np.log(y1)) - np.sum(y2 * np.log(y2))
    )

    
class BackwardElimination:
    def __init__(self, pipeline):
        self.pipe = pipeline
        self._cache = []

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_subset,
        verbose=0,
    ):

        max_iteration_number = len(feature_subset) + 1
        features = feature_subset.copy()
        accuracies = [0.0]
        

        for iteration in range(1, max_iteration_number):
            

            candidate_feature = features[iteration - 1]
            
            if verbose: print(f"Candidate feature to be eliminated is {candidate_feature}")

            feature_subset.remove(candidate_feature)
            

            if verbose: print(f"Remaining features: {feature_subset}")

            since = time.time()

            self.pipe.fit(
                X_train = X_train[feature_subset].values,
                y_train = y_train.values
            )


            score = self.pipe.score(
                X_test = X_test[feature_subset].values,
                y_test = y_test.values
            )

            time_passed = time.time() - since
            accuracies.append(
                score['accuracy']
            )

            self._cache.append(
                (
                    "-".join(
                        feature_subset
                    ),
                    score['accuracy'], 
                    round(float(time_passed), 4)
                )
            )


            if accuracies[iteration] < accuracies[iteration - 1]:
                feature_subset.append(candidate_feature)

                if verbose: print(f"Candidate feature is restored : {candidate_feature}")

        

        self._best_features = feature_subset
        self._scores = accuracies

        return self

    def best_features(self):
        return self._best_features
    
    def cache(self):
        return self._cache

    def scores(self):
        return self._scores
