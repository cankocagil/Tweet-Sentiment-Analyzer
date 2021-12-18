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
    Vocabulary,
    json_print,
    timeit,
    random_seed
)


class KNeighborsClassifier(Classifier):
    """ 
        K-NeighborsClassifier based on geometric distance metrics

    """
    def __init__(
        self,
        k_neighbors:int = 9,
        distance_metric:str = "euclidean"
    ):
        """    
            Args: 

                k_neighbors: int 
                    - Number of Neighbors (Default = 9)
                distance_metric: str
                    - Distance metric (Default = "euclidean") 
                    - Available metrics = [euclidean, manhattan, cosine]
        """   
        super(KNeighborsClassifier, self).__init__()

        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric


        self._hyperparams['k_neighbors'] = self.k_neighbors
        self._hyperparams['distance_metric'] = self.distance_metric
        self._name = 'K-Nearest Neighbors'

    def fit(self, X_train, y_train, *fit_params):
        self.X_train, self.y_train = X_train, y_train

        return self

    def euclidean_distance(self, x1, x2):
        return np.sqrt(
            np.einsum(
                'ij,ij->i...', 
                x1 - x2, 
                x1 - x2
            )
        )

    def manhattan_distance(self, x1, x2):
        return np.linalg.norm(
            x1 - x2,
            axis = 1,
            ord = 1
        )

    def cosine_distance(self, x1, x2):
        y = np.einsum(
            'ij,ij->i',
            x2,
            x2
        )
        x = np.einsum(
            'ij,ij->i',
            x1,
            x1
        )[:, np.newaxis]

        sumxy = x1 @ x2.T
        return 1 - (
            sumxy / np.sqrt(x)
        ) / np.sqrt(y)

        
    def predict(self, X_test):
        
        if self.distance_metric == "euclidean":
            distances = np.array([
                self.euclidean_distance(x_test, self.X_train) for x_test in X_test
            ])
        
        elif self.distance_metric == "manhattan":
            distances = np.array([
                self.manhattan_distance(x_test, self.X_train) for x_test in X_test
            ])

        elif self.distance_metric == "cosine":
            distances = self.cosine_distance(
                X_test,
                self.X_train
            )


        sorted_neighbors = distances.argsort(axis=1)[...,: self.k_neighbors]
        nearest_labels = self.y_train[sorted_neighbors]
        
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis = 1, 
            arr = nearest_labels
        )

        return predictions

class MultiNominalNaiveBayes(Classifier):
    def __init__(self, alpha=0.0001):
        super(MultiNominalNaiveBayes, self).__init__()
        self.alpha = alpha 

        self._hyperparams['alpha'] = self.alpha
        self._name = 'MultiNominal NaiveBayes Classifier'

    @timeit 
    def fit(
        self, 
        X_train:pd.DataFrame,
        y_train: pd.DataFrame,
        **fit_params
    ):  
        
        m, n = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        self.priors = y_train.value_counts(normalize = True).values
        self.counts = pd.concat([X_train, y_train], 1).groupby('class').agg('sum')
        likelihoods = self.counts.T / self.counts.sum(1).values.reshape(-1,  n_classes) + self.alpha
        self.likelihoods = likelihoods.values #.T
        self.log_priors = np.log(self.priors)

        return self
    
    @timeit
    def predict(self, X_test):
        
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        self.log_likelihoods = X_test @ np.log(self.likelihoods) #(np.log(self.likelihoods) @ X_test.T).T 
        self.posteriors = self.log_likelihoods + self.log_priors

        return self.classes[
            self.posteriors.argmax(1)
        ]

    def posteriors(self):
        return self.posteriors

class BernaulliNaiveBayes(Classifier):
    def __init__(self, alpha = 0.001):
        super(BernaulliNaiveBayes, self).__init__()
        self.alpha = alpha 

        self._hyperparams['alpha'] = self.alpha
        self._name = 'Bernaulli NaiveBayes Classifier'

    @timeit 
    def fit(self, X_train, y_train, **fit_params):
        self.classes = np.unique(y_train)

        n_classes = len(self.classes)
        
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        self.priors = y_train.value_counts(normalize = True).values
        self.log_priors = np.log(self.priors)

        counts = pd.concat([X_train, y_train], 1).groupby('class').agg('sum')
        likelihoods = counts.T / counts.sum(1).values.reshape(-1,  n_classes) + self.alpha
        self.likelihoods = likelihoods.T.values

        return self 

    @timeit 
    def predict(self, X_test):

        self.posteriors = np.array(
            [
                (
                    (np.log(self.likelihoods) * x) + (np.log(1 - self.likelihoods) * np.abs(x - 1))
                ).sum(axis = 1) + self.log_priors for x in X_test
            ]
        
        )

        return self.classes[
            self.posteriors.argmax(1)
        ]

class LogisticRegression(Classifier):
    def __init__(self):
        super().__init__()    

    def init_params(self,
        input_shape:int,
        output_shape:int = 1
    ):
        self.__random_seed()

        #assert self.X_train.shape[1] == self.X_test.shape[1], 'Improper feature dimension!'

        W_high = self.__init_xavier(input_shape, output_shape)
        W_low  = - W_high
        W_size = (input_shape, output_shape)
        B_size = (1, output_shape)

        self.W = np.random.uniform(
            W_low,
            W_high,
            size = W_size
        )

        self.b = np.random.uniform(
            W_low,
            W_high, 
            size = B_size
        )


    def __random_seed(self, seed = 32):
        """ Random seed for reproducebility """
        random.seed(seed)
        np.random.seed(seed)

    def __init_xavier(self, L_pre, L_post):
        """ Given the size of the input node and hidden node, initialize the weights drawn from uniform distribution ~ Uniform[- sqrt(6/(L_pre + L_post)) , sqrt(6/(L_pre + L_post))]  """    
        return np.sqrt(6/(L_pre + L_post))   

    def __train_config(self,
        lr:float,
        batch_size:int,
        epochs:int,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def sigmoid(self,X, grad = False):
        """ Computing sigmoid and it's gradient w.r.t. it's input """
        sig = 1/(1 + np.exp(-X))

        return sig * (1-sig) if grad else sig

    def __forward(self, X):

        Z = (X @ self.W)  + self.b
        A = self.sigmoid(Z)

        return {
            "Z": Z,
            "A": A
        }

    
    def __SGD(self, grads):
        self.W -= self.lr * grads['W']  
        self.b -= self.lr * grads['b']
    

    def matrix_back_prop(self, outs, X, Y):
        """ Matrix form backward propagation """
        m = self.batch_size 
            
        Z = outs['Z']
        A = outs['A']

        dZ = (A-Y) * self.sigmoid(Z, grad = True)    
        dW = (1 / m) * (X.T @ dZ)  
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True) 
        
        assert self.W.shape == dW.shape, f'Error in weight shapes!, {dW.shape} does not match with {self.W.shape}'
        assert self.b.shape == db.shape, f'Error in bias shapes!, {db.shape} does not match with {self.b.shape}'

        grads = {}
        grads['W'] = dW
        grads['b'] = db

        return grads


    def backward(self,
        outs,
        X,
        Y
    ):
        return self.matrix_back_prop(
            outs,
            X,
            Y
        )

    
    def BinaryCrossEntropyLoss(self, pred, label):        
        m = pred.shape[0]
        preds = np.clip(pred, 1e-16, 1 - 1e-16)
        loss = np.sum(-label * np.log(preds + 1e-20) - (1 - label) * np.log(1 - preds + 1e-20))        
        return loss / m 
    
    def eval(self, x, y, knob:float = 0.5):
        predictions = self.__forward(x)
        predictions = predictions['A']
        predictions[predictions>=knob] = 1
        predictions[predictions< knob] = 0
        acc_score = self.accuracy(predictions, y)

        return acc_score

    def __accuracy(self,pred,label):
        return np.sum(pred == label) / pred.shape[0]


    @timeit
    def fit(
        self, 
        X_train,
        y_train,
        X_test,
        y_test, 
        lr:float = 1e-2, 
        batch_size:int = 32,
        epochs:int = 100, 
        verbose = True
    ):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """

        self.init_params(
            input_shape = X_train.shape[1]
            )

        self.__train_config(
            lr,
            batch_size,
            epochs,
        )

        self.history = {}

        self.history['train'] = {
            'loss': [],
            'acc' : []
        }

        self.history['val'] =   {
            'loss': [],
            'acc' : []
        }
        
        m = self.batch_size
        
        self.sample_size_train = X_train.shape[0]
        
        for epoch in range(self.epochs):

            perm = np.random.permutation(self.sample_size_train)          
            
            for i in range(self.sample_size_train // m):
                

                shuffled_index = perm[i*m: (i+1)*m]
                
                X_feed = X_train[shuffled_index]    
                y_feed = y_train[shuffled_index]
                                
                outs = self.__forward(X_feed)              
                grads = self.backward(
                    outs,
                    X_feed,
                    y_feed
                )
                self.__SGD(grads)

            loss_train = self.BinaryCrossEntropyLoss(
                self.__forward(X_train)['A'],
                y_train
            )     

            acc_train = self.eval(
                X_train,
                y_train
            )   

            self.history['train']['loss'].append(loss_train)
            self.history['train']['acc'].append(acc_train)
                
            
                
            loss_val = self.BinaryCrossEntropyLoss(
                self.__forward(X_test)['A'],
                y_test
            )     

            acc_val  = self.eval(
                X_test,
                y_test
            )             

            self.history['val']['loss'].append(loss_val)
            self.history['val']['acc'].append(acc_val)

            if verbose:                    
                print(f"[{epoch}/{self.epochs}] ------> Training : BCE: {loss_train} and Acc: {acc_train}")                       
                print(f"[{epoch}/{self.epochs}] ------> Testing  : BCE: {loss_val}   and  Acc: {acc_val}")


    def __str__(self):
        model = LogisticRegression().__class__.__name__
        model += f' with hyperparameters (learning rate,batch_size,epochs) = ({self.lr,self.batch_size,self.epochs})'
        num_params = self.W.shape[0] * self.W.shape[1] + self.b.shape[0] * self.b.shape[1]
        model += f'\n There are {num_params} number of traniable parameters'
        return model

    def __repr__(self):
        model = LogisticRegression().__class__.__name__    
        model += f' with hyperparameters (learning rate,batch_size,epochs) = ({self.lr,self.batch_size,self.epochs})'
        num_params = self.W.shape[0] * self.W.shape[1] + self.b.shape[0] * self.b.shape[1]
        model += f'\n There are {num_params} number of traniable parameters'
        return model

    def plot_history(self):

        fig,axs = plt.subplots(1,2,figsize = (24,8))
        axs[0].plot(self.history['train']['loss'],color = 'orange',label = 'Training')
        axs[0].plot(self.history['val']['loss'], label = 'Validation')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('BCE Loss')
        axs[0].set_title(f'Binary Cross Entropy Loss Over Iterations with Learning Rate $\eta$ = {self.lr}')
        axs[0].legend(loc="upper right")  
        axs[0].grid()

        axs[1].plot(self.history['train']['acc'],color ='orange',label = 'Training')
        axs[1].plot(self.history['val']['acc'], label = 'Validation')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        maxs = round(max(self.history['train']['acc']),3), round(max(self.history['val']['acc']),3)
        axs[1].set_title(f'Accuracy Over Iterations with Learning Rate $\eta$ =  {self.lr} \n Best Accuracy in (Training,Validation) = {maxs} ')
        axs[1].legend(loc="lower right")   
        axs[1].grid()


class MLP(Classifier):
    
    def __init__(self,
        input_size = X_train.shape,
        batch_size = 19 ,
        n_neurons = 76 , 
        mean = 0,
        std = 1,
        lr = 1e-1,
        distribution = 'Xavier'
    ):


        np.random.seed(15)
        self.lr = lr                              
        self.mse_train = {}
        self.mce_train = {}
        self.mse_test = {}
        self.mce_test = {}
        
        self.sample_size = input_size[0]
        self.feature_size = input_size[1]
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.mean, self.std = mean, std
        
        self.dist = distribution
        
        
        self.n_update = round((self.sample_size/self.batch_size))
        
        self.W1_size = self.feature_size,self.n_neurons        
        self.W2_size = self.n_neurons,1
        
        self.B1_size = 1,self.n_neurons
        self.B2_size = 1, 1
        
        self.B1 = np.random.normal(loc = self.mean, scale = self.std, size = (self.B1_size)) * 0.01
        self.B2 = np.random.normal(loc = self.mean, scale = self.std, size = (self.B2_size)) * 0.01  
        
        self.he_scale1 = np.sqrt(2/self.feature_size)
        self.he_scale2 = np.sqrt(2/self.n_neurons)
        self.xavier_scale1 = np.sqrt(2/(self.feature_size+self.n_neurons))
        self.xavier_scale2 = np.sqrt(2/(self.n_neurons+1))      
        
        if (self.dist == 'Zero') :
            self.W1 = np.zeros((self.W1_size))
            self.W2 = np.zeros((self.W2_size))

        elif (self.dist == 'Gauss'):
            self.W1 = np.random.normal(loc = self.mean, scale = self.std, size = (self.W1_size))* 0.01
            self.W2 = np.random.normal(loc = self.mean, scale = self.std, size = (self.W2_size))* 0.01

        elif (self.dist == 'He'):
            self.W1 = np.random.randn(self.W1_size[0],self.W1_size[1]) * self.he_scale1
            self.W2 = np.random.randn(self.W2_size[0],self.W2_size[1]) * self.he_scale2

        elif (self.dist == 'Xavier'):

            self.W1 = np.random.randn(self.W1_size[0],self.W1_size[1]) * self.xavier_scale1
            self.W2 = np.random.randn(self.W2_size[0],self.W2_size[1]) * self.xavier_scale2 
            

        
    def forward(self,X):        
        
        Z1 = (X @ self.W1)  + self.B1 
        A1 = np.tanh(Z1)        
        Z2 = (A1 @ self.W2) + self.B2
        A2 = np.tanh(Z2)
               
        return {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }
    
    
    def tanh(self,X):        
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

    def tanh_der(self,X):        
        return  1-(np.tanh(X)**2)
    
    def backward(self,outs, X, Y):
        m = (self.batch_size) 
        
        Z1 = outs['Z1']
        A1 = outs['A1']
        Z2 = outs['Z2']
        A2 = outs['A2']
        
        dZ2 = (A2-Y)* self.tanh_der(Z2)
        dW2 = (1/m) * (A1.T @ dZ2)
        dB2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True) 
        
        dZ1 = (dZ2 @ self.W2.T) * self.tanh_der(Z1)
        dW1 = (1/m) * (X.T @ dZ1)
        dB1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        
        return {
            "dW1": dW1,
            "dW2": dW2,
            "dB1": dB1,
            "dB2": dB2
        }  
        
    def Loss(self,pred, y_true, knob = 0):
        
        mse = np.square(pred-y_true).mean()
        
        pred[pred>=knob]=1 
        pred[pred<knob]=-1         
        
        mce = (pred == y_true).mean()
        
        return {
            'MSE':mse,
            'MCE':mce
        }
      
       
    def SGD(self,grads):        
        self.W1 -= self.lr * grads['dW1']     
        self.W2 -= self.lr * grads['dW2']
        self.B1 -= self.lr * grads['dB1']     
        self.B2 -= self.lr * grads['dB2']
        
    def fit(self,X,Y,X_test,y_test,epochs = 300,verbose=True):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
        
        m = self.batch_size
        
        for epoch in range(epochs):
            perm = np.random.permutation(self.sample_size)          
            
            for i in range(self.n_update):
                
                
                batch_start = i * m
                batch_finish = (i+1) * m
                index = perm[batch_start:batch_finish]
                
                X_feed = X[index]    
                y_feed = Y[index]
                
                
                outs = self.forward(X_feed)              
                loss = self.Loss(
                    outs['A2'],
                    y_feed
                )
                                
                outs_test = self.forward(X_test)              
                loss_test = self.Loss(
                    outs_test['A2'],
                    y_test
                )                                

                grads = self.backward(
                    outs,
                    X_feed,
                    y_feed
                )     
        
                self.SGD(grads)
                
            self.mse_train[f"Epoch:{epoch}"] = loss['MSE']
            self.mce_train[f"Epoch:{epoch}"] = loss['MCE']
            self.mse_test[f"Epoch:{epoch}"] = loss_test['MSE']
            self.mce_test[f"Epoch:{epoch}"] = loss_test['MCE']

            if verbose:                    
                print(f"[{epoch}/{epochs}] ------> Training :MSE: {loss['MSE']} and MCE: {loss['MCE']}")                        
                print(f"[{epoch}/{epochs}] ------> Testing :MSE: {loss_test['MSE']} and MCE: {loss_test['MCE']}")
        
    def history(self):      
        return {
            'Train_MSE' : self.mse_train,
            'Train_MCE' : self.mce_train,
            'Test_MSE'  : self.mse_test,
            'Test_MCE'  : self.mce_test
        }