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
    

def save_obj(
    
    obj:object,
    path:str = None
) -> None:
    """ Saves Python Object as pickle"""
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(
    path:str = None
) -> object:
    """ Loads Python Object from pickle"""
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_numpy(
    data:Iterable[list or np.ndarray] = None,
    path:str = None
) -> None:
    """ Saves NumPy array or Python object as .npy"""
    np.save(
        path + '.npy',
        data, 
        allow_pickle=True
    )

def load_numpy(
    path:str = None
) -> np.ndarray:
    """ Loads NumPy array or Python object as .npy"""
    return np.load(
        path + '.npy',
        allow_pickle=True
        )

class ClassifierCharacteristics:

    def confusion_matrix(
        self,
        preds: Iterable[list or np.ndarray],
        labels: Iterable[list or np.ndarray]
    ) -> pd.DataFrame:
        
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

    def accuracy(
        self,
        preds: Iterable[list or np.ndarray],
        labels: Iterable[list or np.ndarray],
        scale:bool = True
    ) -> np.float:
        """Given the labels and predictions, calculate accuracy score. """
        return np.mean(preds == labels) * 100 if scale else np.mean(preds == labels)

    def visualize_confusion_matrix(
        self,
        data:Iterable[list or np.ndarray],
        normalize:bool = True,
        title:str = " ", 
    ) -> None:
        
        if normalize:
            data /= np.sum(data)

        plt.figure(
            figsize = (
                15, 15
            )
        )
        sns.heatmap(
            data, 
            fmt='.2%',
            cmap = 'Greens'
        )

        plt.title(title)
        plt.show()

    @staticmethod
    def timeit(
        Func:Callable
    ):
        """ Calculate time spend of the function
        
        Usage:
            >>  @timeit
            >>  def func(x):
            >>     return x
        """
        def _timeStamp(*args, **kwargs):
            since = time.time()
            result = Func(*args, **kwargs)
            time_elapsed = time.time() - since

            if time_elapsed > 60:
                print('Time Consumed : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
            else:        
                print('Time Consumed : ' , round((time_elapsed), 4) , 's')
            return result
        return _timeStamp

    @staticmethod
    def random_seed(
        Func:Callable,
        seed:int = 42
    ):
        """
        Decorator random seed.
            
        Usage:
            >>  @random_seed
            >>  def func(*args):
            >>     return [arg for arg in args]
        """
        def _random_seed(*args, **kwargs):
            np.random.seed(seed)
            random.seed(seed)
            result = Func(
                *args,
                **kwargs
            )
            return result
        return _random_seed
        
    def save_obj(
        self,
        obj:object,
        path:str = None
    ) -> None:
        """ Saves Python Object as pickle"""
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def load_obj(
        self,
        path:str = None
    ) -> object:
        """ Loads Python Object from pickle"""
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_numpy(
        self,
        data:Iterable[list or np.ndarray] = None,
        path:str = None
    ) -> None:
        """ Saves NumPy array or Python object as .npy"""
        np.save(
            path + '.npy',
            data, 
            allow_pickle=True
        )

    def load_numpy(
        self,
        path:str = None
    ) -> np.ndarray:
        """ Loads NumPy array or Python object as .npy"""
        return np.load(
            path + '.npy',
            allow_pickle=True
        )


    def save_obj(
        self,
        obj:object,
        path:str = None
    ) -> None:
        """ Saves Python Object as pickle"""
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class Classifier(ClassifierCharacteristics):
    #__metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Classifier, self).__init__()
        self._hyperparams = {}
        self._scores = {}
        self._name = ""
        self._params = OrderedDict()


    def save(self, filename: str) -> None:
        self.save_obj(
            self.__dict__,
            filename
        )

    def load(self, filename: str) -> None:
        self.__dict__ = self.load_obj(filename)


    @ClassifierCharacteristics.timeit
    def fit(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        *fit_params
    ) -> None:
        return NotImplementedError()
    
    

    @ClassifierCharacteristics.timeit
    def predict(
        self, 
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return NotImplementedError()



    @ClassifierCharacteristics.timeit
    def fit_predict(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        *fit_params
    ) -> Union[pd.DataFrame, np.ndarray]:

        self.fit(X_train, y_train, *fit_params)

        return self.predict(X_test)

    def score(
        self, 
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame],
        metric_list: List[Callable] = []
    ) -> np.float:

        predictions = self.predict(X_test)
        accuracy = self.accuracy(predictions, y_test)


        self._scores['accuracy'] = accuracy

        if len(metric_list) != 0:
            for metric in metric_list:
                self._scores[metric.__name__] = metric(y_test, predictions)

        return self._scores

  
    def params(self) -> OrderedDict:
        return self._params
        

    def hyperparameters(self):
        return self._hyperparams

 
    def name(self) -> str:
        return self._name 

  
    def __str__(self)-> str:
        return f"{self._name} with hyperparameters {json.dumps(self._hyperparams, sort_keys=True, indent=4)}"

  
    def __repr__(self)-> str:
        return f"{self._name} with hyperparameters {json.dumps(self._hyperparams, sort_keys=True, indent=4)}"


class Pipeline:
    """ Generic ML Operation Pipeline """
    def __init__(
        self,
        pipeline: List[Tuple[str, object]] = []
    ):
        super(Pipeline, self).__init__()
        self.pipeline = pipeline
        self.model = None
        self._name = 'ML Pipeline'
        self._scores = {}

    def fit(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        verbose: int = 0,
        *fit_params
    ) -> None:

        yield_data = X_train.copy()

        for pipe_name, pipe_op in self.pipeline:
            
            if verbose: print(f"{pipe_name} operation is applying")

            if issubclass(pipe_op.__class__, Classifier):

                self.model = pipe_op
                self.model.fit(
                    yield_data,
                    y_train,
                    *fit_params
                )
            
            elif isinstance(pipe_op.__class__, FeatureEngineer): 
                yield_data = pipe_op.fit_transform(yield_data)

            else: 
                raise Exception(f"{pipe_name} operator could not be decoded!")


    def predict(
        self, 
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Union[pd.DataFrame, np.ndarray]:
        return self.model.predict(X_test)


    def score(
        self, 
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame],
        metric_list: List[Callable] = []
    ) -> np.float:

        predictions = self.model.predict(X_test)
        accuracy = self.model.accuracy(predictions, y_test)


        self._scores['accuracy'] = accuracy

        if len(metric_list) != 0:
            for metric in metric_list:
                self._scores[metric.__name__] = metric(predictions, y_test)

        return self._scores
 
    def name(self) -> str:
        return self._name 

  
    def __str__(self) -> str:
        return "\n".join([
            str(pape_op) for _, pape_op in self.pipeline
        ])
    def __repr__(self) -> str:
        return "\n".join([
            str(pape_op) for _, pape_op in self.pipeline
        ])



class Vocabulary:
    def __init__(
        self,
        root_dir:str,
        filename:str,
        delimiter: str = '\n'
        
    ):
        super(Vocabulary, self).__init__()
        self.vocab = open(
            os.path.join(
                root_dir,
                filename
            )
        ).read()

        self.list_vocab = self.vocab.split(delimiter)[:-1]

        self.word2id = {
            word: i for i, word in enumerate(self.list_vocab)
        } 
        
        self.id2word = {
            i: word for word, i in self.word2id.items()
        }

    def __getitem__(self, idx):

        if isinstance(idx, (list, np.ndarray)):
            return [self.id2word[i] for i in idx]

        return self.id2word[idx]

    def __len__(self):
        return len(self.list_vocab)

    def get_vocab(self):
        return self.word2id


def json_print(data):
    return json.dumps(
        data,
        sort_keys=True,
        indent=4
    )



def timeit(
    Func:Callable
):
    """ Calculate time spend of the function
    
    Usage:
        >>  @timeit
        >>  def func(x):
        >>     return x
    """
    def _timeStamp(*args, **kwargs):
        since = time.time()
        result = Func(*args, **kwargs)
        time_elapsed = time.time() - since

        if time_elapsed > 60:
            print('Time Consumed for {}: {:.0f}m {:.0f}s'.format(Func.__name__, time_elapsed // 60, time_elapsed % 60))  
        else:        
            print(f'Time Consumed for {Func.__name__}: {round((time_elapsed), 4)} s')
        return result
    return _timeStamp

def random_seed(
    Func:Callable,
    seed:int = 42
):
    """
    Decorator random seed.
        
    Usage:
        >>  @random_seed
        >>  def func(*args):
        >>     return [arg for arg in args]
    """
    def _random_seed(*args, **kwargs):
        np.random.seed(seed)
        random.seed(seed)
        result = Func(
            *args,
            **kwargs
        )
        return result
    return _random_seed