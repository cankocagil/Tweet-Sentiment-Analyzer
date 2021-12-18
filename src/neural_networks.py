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


class RNN(object):
    """ Recurrent Neural Network (RNN). RNN encapsulates all necessary logic for training the network. """
    
    def __init__(
        self,
        input_dim = 3,
        hidden_dim = 128, 
        seq_len = 150, 
        learning_rate = 1e-1, 
        mom_coeff = 0.85, 
        batch_size = 32, 
        output_class = 6
    ):

        """ Initialization of weights/biases and other configurable parameters.  """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim_inp2hid = Xavier(self.input_dim,self.hidden_dim)
        self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim))
        self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim))

        lim_hid2hid = Xavier(self.hidden_dim,self.hidden_dim)
        self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim,self.hidden_dim))

        lim_hid2out = Xavier(self.hidden_dim,self.output_class)
        self.W2 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim,self.output_class))
        self.B2 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss, self.test_loss, self.train_acc, self.test_acc = [],[],[],[]
        
        # Storing previous momentum updates :
        self.prev_updates = {
            'W1'       : 0,
            'B1'       : 0,
            'W1_rec'   : 0,
            'W2'       : 0,
            'B2'       : 0
        }


    def forward(self, X) -> tuple:
        """ 
        Forward propagation of the RNN through time.
        
        * X_state is the input across all time steps
        * hidden_state is the hidden stages across time
        * probs is the probabilities of each outputs, i.e. outputs of softmax.
        
        Returns:
            * (X_state, hidden_state, probs) as a tuple.       
        """ 
        X_state = dict()
        hidden_state = dict()
        output_state = dict()
        probs = dict()

        self.h_prev_state = np.zeros((1,self.hidden_dim))
        hidden_state[-1] = np.copy(self.h_prev_state)

        # Loop over time T:
        for t in range(self.seq_len):

            # Selecting first record with inputs, dimension = (batch_size, input_size)
            X_state[t] = X[:, t]

            # Recurrent hidden layer :
            hidden_state[t] = np.tanh(
                np.dot(X_state[t], self.W1) + np.dot(hidden_state[t-1], self.W1_rec) + self.B1
            )

            output_state[t] = np.dot(
                hidden_state[t], self.W2
            ) + self.B2

            # Per class probabilites :
            probs[t] = activations.softmax(output_state[t], axis=-1)

        return X_state, hidden_state, probs
        

    def BPTT(self, cache, Y):
        """
        Back propagation through time algorihm.
        
        Inputs:
         * Cache = (X_state, hidden_state, probs)
         * Y = desired output

        Returns:
            * Gradients w.r.t. all configurable elements
        """

        X_state, hidden_state, probs = cache

        # backward pass: compute gradients going backwards
        dW1, dW1_rec, dW2 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2)

        dB1, dB2 = np.zeros_like(self.B1), np.zeros_like(self.B2)

        dhnext = np.zeros_like(hidden_state[0])

        dy = np.copy(probs[self.seq_len - 1])      
        dy[np.arange(len(Y)), np.argmax(Y, axis = 1)] -= 1
        
        dB2 += np.sum(dy, axis = 0, keepdims = True)
        dW2 += np.dot(hidden_state[self.seq_len - 1].T, dy)

        for t in reversed(range(1, self.seq_len)):


            dh = np.dot(dy, self.W2.T) + dhnext
        
            dhrec = (1 - (hidden_state[t] * hidden_state[t])) * dh

            dB1 += np.sum(dhrec, axis = 0, keepdims = True)
            
            dW1 += np.dot(X_state[t].T, dhrec)
            
            dW1_rec += np.dot(hidden_state[t-1].T, dhrec)

            dhnext = np.dot(dhrec, self.W1_rec.T)


        for grad in [dW1,dB1,dW1_rec,dW2,dB2]:
            np.clip(grad, -10, 10, out = grad)


        return [dW1, dB1, dW1_rec, dW2, dB2]    
        
    def earlyStopping(self, ce_train, ce_val, ce_threshold, acc_train, acc_val, acc_threshold):
        return any([
            ce_train - ce_val < ce_threshold, 
            acc_train - acc_val > acc_threshold
        ])
    
    def CategoricalCrossEntropy(self, labels, preds):
        """ Computes cross entropy between labels and model's predictions """
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N

    def step(self, grads, momentum = True):
        """ SGD w/o Momentum on mini batches """

        if momentum:
            
            delta_W1 = -self.learning_rate * grads[0] +  self.mom_coeff * self.prev_updates['W1']
            delta_B1 = -self.learning_rate * grads[1] +  self.mom_coeff * self.prev_updates['B1']  
            delta_W1_rec = -self.learning_rate * grads[2] +  self.mom_coeff * self.prev_updates['W1_rec']
            delta_W2 = -self.learning_rate * grads[3] +  self.mom_coeff * self.prev_updates['W2']              
            delta_B2 = -self.learning_rate * grads[4] +  self.mom_coeff * self.prev_updates['B2']
            
            self.W1 += delta_W1
            self.W1_rec += delta_W1_rec
            self.W2 += delta_W2
            self.B1 += delta_B1
            self.B2 += delta_B2     

            self.prev_updates['W1'] = delta_W1
            self.prev_updates['W1_rec'] = delta_W1_rec
            self.prev_updates['W2'] = delta_W2
            self.prev_updates['B1'] = delta_B1
            self.prev_updates['B2'] = delta_B2

            self.learning_rate *= 0.9999

    def fit(self, X, Y, X_val, y_val, epochs = 50, verbose = True, earlystopping = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(X.shape[0])           
            
            for i in range(round(X.shape[0]/ self.batch_size)): 

                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size
                index = perm[batch_start:batch_finish]
                
                X_feed = X[index]    
                y_feed = Y[index]
                
                cache_train = self.forward(X_feed)                                                          
                grads = self.BPTT(cache_train, y_feed)                
                self.step(grads)

            cross_loss_train = self.CategoricalCrossEntropy(y_feed, cache_train[2][self.seq_len - 1])
            predictions_train = self.predict(X)
            acc_train = accuracy(
                np.argmax(Y, axis = 1),
                predictions_train
            )

            _, __, probs_test = self.forward(X_val)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[self.seq_len - 1])
            predictions_val = np.argmax(probs_test[self.seq_len - 1], 1)
            acc_val = accuracy(
                np.argmax(y_val, axis = 1),
                predictions_val
            )
            

            if earlystopping:                
                if self.earlyStopping(
                    ce_train = cross_loss_train, 
                    ce_val = cross_loss_val, 
                    ce_threshold = 3.0,
                    acc_train = acc_train,
                    acc_val = acc_val,
                    acc_threshold = 15
                ): 
                    break

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)

    def predict(self,X):
        _, __, probs = self.forward(X)
        return np.argmax(
            probs[self.seq_len - 1],
            axis=1
        )

    def history(self):
        return {
            'TrainLoss' : self.train_loss,
            'TrainAcc'  : self.train_acc,
            'TestLoss'  : self.test_loss,
            'TestAcc'   : self.test_acc
        }

class LSTM:
    """ Long-Short Term Memory Recurrent neural network, encapsulates all necessary logic for training,
        then built the hyperparameters and architecture of the network.
    """

    def __init__(
      self,
      input_dim = 3,
      hidden_dim = 100,
      output_class = 6,
      seq_len = 150,
      batch_size = 30,
      learning_rate = 1e-1,
      mom_coeff = 0.85,
      random_state = 150
    ):
        """ Initialization of weights/biases and other configurable parameters. """
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Unfold case T:
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        self.input_stack_dim = self.input_dim + self.hidden_dim
        
        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6 / (fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim)
        self.W_f = np.random.uniform(-lim1,lim1,(self.input_stack_dim, self.hidden_dim))
        self.B_f = np.random.uniform(-lim1,lim1,(1, self.hidden_dim))

        self.W_i = np.random.uniform(-lim1,lim1,(self.input_stack_dim, self.hidden_dim))
        self.B_i = np.random.uniform(-lim1,lim1,(1, self.hidden_dim))

        self.W_c = np.random.uniform(-lim1,lim1,(self.input_stack_dim, self.hidden_dim))
        self.B_c = np.random.uniform(-lim1,lim1,(1, self.hidden_dim))

        self.W_o = np.random.uniform(-lim1,lim1,(self.input_stack_dim, self.hidden_dim))
        self.B_o = np.random.uniform(-lim1,lim1,(1, self.hidden_dim))
        
        lim2 = Xavier(self.hidden_dim,self.output_class)
        self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim, self.output_class))
        self.B = np.random.uniform(-lim2,lim2,(1, self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss, self.test_loss, self.train_acc, self.test_acc = [],[],[],[]
        
        # To keep previous updates in momentum :
        self.previous_updates = [0] * 10
        
        # For AdaGrad:
        self.cache = [0] * 10     
        self.cache_rmsprop = [0] * 10
        self.m = [0] * 10
        self.v = [0] * 10
        self.t = 1

    def cell_forward(self,X, h_prev, C_prev):
        """
        Takes input, previous hidden state and previous cell state, compute:
          * Forget gate + Input gate + New candidate input + New cell state + 
          * output gate + hidden state. Then, classify by softmax.
        """

        # Stacking previous hidden state vector with inputs:
        stack = np.column_stack([X, h_prev])

        # Forget gate:
        forget_gate = activations.sigmoid(
          np.dot(stack, self.W_f) + self.B_f
        )
  
        # İnput gate:
        input_gate = activations.sigmoid(
          np.dot(stack, self.W_i) + self.B_i
        )

        # New candidate:
        cell_bar = np.tanh(
          np.dot(stack, self.W_c) + self.B_c
        )

        # New Cell state:
        cell_state = forget_gate * C_prev + input_gate * cell_bar

        # Output fate:
        output_gate = activations.sigmoid(
          np.dot(stack, self.W_o) + self.B_o
        )

        # Hidden state:
        hidden_state = output_gate * np.tanh(cell_state)

        # Classifiers (Softmax) :
        dense = np.dot(hidden_state, self.W) + self.B
        probs = activations.softmax(dense, axis=-1)

        return (stack, forget_gate, input_gate, cell_bar, cell_state, output_gate, hidden_state, dense, probs)

    def forward(self, X, h_prev, C_prev):
        x_s, z_s, f_s, i_s = {}, {}, {}, {}
        C_bar_s, C_s, o_s, h_s = {}, {}, {},{}
        v_s, y_s = {}, {}


        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]

            z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t] = self.cell_forward(
              x_s[t],
              h_s[t-1],
              C_s[t-1]
            )

        return (z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s)
    
    def BPTT(self, outs, Y):

        z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s = outs

        dW_f, dW_i, dW_c, dW_o, dW = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c), np.zeros_like(self.W_o), np.zeros_like(self.W)

        dB_f, dB_i, dB_c, dB_o, dB = np.zeros_like(self.B_f), np.zeros_like(self.B_i), np.zeros_like(self.B_c), np.zeros_like(self.B_o), np.zeros_like(self.B)

        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0])   

        # w.r.t. softmax input
        ddense = np.copy(y_s[self.seq_len - 1])
        ddense[np.arange(len(Y)), np.argmax(Y, axis=1)] -= 1
        #ddense[np.argmax(Y,1)] -=1
        #ddense = y_s[149] - Y
        # Softmax classifier's :
        dW = np.dot(h_s[self.seq_len - 1].T,ddense)
        dB = np.sum(ddense, axis = 0, keepdims = True)

        # Backprop through time:
        for t in reversed(range(1, self.seq_len)):           
            
            # Just equating more meaningful names
            stack, forget_gate, input_gate, cell_bar, cell_state, output_gate, hidden_state, dense, probs = z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]
            C_prev = C_s[t-1]
            
            # w.r.t. softmax input
            #ddense = np.copy(probs)
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
            # Softmax classifier's :
            #dW += np.dot(hidden_state.T,ddense)
            #dB += np.sum(ddense,axis = 0, keepdims = True)

            # Output gate :
            dh = np.dot(ddense, self.W.T) + dh_next            
            do = dh * np.tanh(cell_state)
            do = do * dsigmoid(output_gate)
            dW_o += np.dot(stack.T,do)
            dB_o += np.sum(do, axis = 0, keepdims = True)

            # Cell state:
            dC = np.copy(dC_next)
            dC += dh * output_gate * activations.dtanh(cell_state)
            dC_bar = dC * input_gate
            dC_bar = dC_bar * dtanh(cell_bar) 
            dW_c += np.dot(stack.T, dC_bar)
            dB_c += np.sum(dC_bar,axis = 0, keepdims = True)
            
            # Input gate:
            di = dC * cell_bar
            di = dsigmoid(input_gate) * di
            dW_i += np.dot(stack.T, di)
            dB_i += np.sum(di, axis = 0, keepdims = True)

            # Forget gate:
            df = dC * C_prev
            df = df * dsigmoid(forget_gate) 
            dW_f += np.dot(stack.T, df)
            dB_f += np.sum(df, axis = 0, keepdims = True)

            dz = np.dot(df, self.W_f.T) + np.dot(di, self.W_i.T) + np.dot(dC_bar, self.W_c.T) + np.dot(do, self.W_o.T)

            dh_next = dz[:, -self.hidden_dim:]
            dC_next = forget_gate * dC
        
        # List of gradients :
        grads = [dW, dB, dW_o, dB_o, dW_c, dB_c, dW_i, dB_i, dW_f, dB_f]

        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1], C_s[self.seq_len -1 ], grads
    


    def fit(self, X, Y, X_val, y_val, epochs = 50, optimizer = 'SGD', verbose = True, crossVal = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(X.shape[0])           
            h_prev, C_prev = np.zeros((self.batch_size, self.hidden_dim)), np.zeros((self.batch_size, self.hidden_dim))
            
            for i in range(round(X.shape[0] / self.batch_size) - 1): 
          
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
          
                # Forward + BPTT + SGD:
                cache_train = self.forward(X_feed, h_prev, C_prev)
                h,c,grads = self.BPTT(cache_train, y_feed)

                if optimizer == 'SGD':                                                                        
                  self.SGD(grads)

                elif optimizer == 'AdaGrad' :
                  self.AdaGrad(grads)

                elif optimizer == 'RMSprop':
                  self.RMSprop(grads)
                
                elif optimizer == 'VanillaAdam':
                  self.VanillaAdam(grads)
                else:
                  self.Adam(grads)

                # Hidden state -------> Previous hidden state
                # Cell state ---------> Previous cell state
                h_prev, C_prev = h, c

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed, cache_train[8][self.seq_len - 1])
            predictions_train = self.predict(X)
            acc_train = accuracy(
              np.argmax(Y, axis=1),
              predictions_train
            )

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0], self.hidden_dim))
            _,__,___,____,_____,______,_______,________, probs_test = self.forward(X_val, test_prevs, test_prevs)
            cross_loss_val = self.CategoricalCrossEntropy(y_val, probs_test[self.seq_len - 1])
            
            predictions_val = np.argmax(
              probs_test[self.seq_len - 1],
              axis = 1
            )

            acc_val = accuracy(
              np.argmax(y_val,axis=1), 
              predictions_val
            )

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)
      
    
    def params(self):
        """
        Return all weights/biases in sequential order starting from end in list form.

        """        
        return [self.W, self.B, self.W_o, self.B_o, self.W_c, self.B_c, self.W_i, self.B_i, self.W_f, self.B_f]


    def SGD(self,grads):
      """ Stochastic gradient descent with momentum on mini-batches. """
      prevs = []
      for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
          delta = self.learning_rate * grad - self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)

      self.previous_updates = prevs       

      self.learning_rate *= 0.99999   

    
    def AdaGrad(self, grads):
      """ AdaGrad adaptive optimization algorithm. """         

      for i, (param, grad) in enumerate(zip(self.params(), grads)):
        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)


    def RMSprop(self, grads, decay_rate = 0.9):
      """ RMSprop adaptive optimization algorithm """

      for i, (param, grad) in enumerate(zip(self.params(), grads)):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        


    def VanillaAdam(self, grads, beta1 = 0.9, beta2 = 0.999):
        """ Adam optimizer, but bias correction is not implemented """
      
        for i, (param, grad)  in enumerate(zip(self.params(), grads)):
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)


    def Adam(self, grads, beta1 = 0.9, beta2 = 0.999):
        """ Adam optimizer, bias correction is implemented.
        """

        for i, (param, grad) in enumerate(zip(self.params(), grads)):
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)

        self.t +=1
    
    
    def CategoricalCrossEntropy(self,labels,preds):
        """ Computes cross entropy between labels and model's predictions """
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N
    
    def predict(self,X):
        """ Return predictions, (not one hot encoded format) """

        # Give zeros to hidden/cell states:
        pasts = np.zeros((X.shape[0], self.hidden_dim))
        _, __ ,___ ,____, _____, ______, _______, _______, probs = self.forward(X, pasts, pasts)
        return np.argmax(probs[self.seq_len - 1], axis=1)

    def history(self):
        return {
          'TrainLoss' : self.train_loss,
          'TrainAcc'  : self.train_acc,
          'TestLoss'  : self.test_loss,
          'TestAcc'   : self.test_acc
        }  


class GRU:
    """
    Gater recurrent unit, encapsulates all necessary logic for training, 
    then built the hyperparameters and architecture of the network.
    """

    def __init__(
      self,
      input_dim = 3,
      hidden_dim = 128,
      output_class = 6,
      seq_len = 150,
      batch_size = 32,
      learning_rate = 1e-1,
      mom_coeff = 0.85,
      random_state = 32
    ):
        """ Initialization of weights/biases and other configurable parameters. """
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim)
        lim1_hid = Xavier(self.hidden_dim,self.hidden_dim)
        self.W_z = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
        self.U_z = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
        self.B_z = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

        self.W_r = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
        self.U_r = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
        self.B_r = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

        self.W_h = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim))
        self.U_h = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim,self.hidden_dim))
        self.B_h = np.random.uniform(-lim1,lim1,(1,self.hidden_dim))

        
        lim2 = Xavier(self.hidden_dim,self.output_class)
        self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim,self.output_class))
        self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss, self.test_loss, self.train_acc, self.test_acc = [],[],[],[]
        
        # To keep previous updates in momentum :
        self.previous_updates = [0] * 10
        
        # For AdaGrad:
        self.cache = [0] * 11   
        self.cache_rmsprop = [0] * 11
        self.m = [0] * 11
        self.v = [0] * 11
        self.t = 1

    def cell_forward(self,X,h_prev):
        """

        Takes input, previous hidden state and previous cell state, compute:
          * Forget gate + Input gate + New candidate input + New cell state + 
          * output gate + hidden state. Then, classify by softmax.
        """
                      

        # Update gate:
        update_gate = activations.sigmoid(
          np.dot(X, self.W_z) + np.dot(h_prev, self.U_z) + self.B_z
        )
       
        # Reset gate:
        reset_gate = activations.sigmoid(
          np.dot(X, self.W_r) + np.dot(h_prev, self.U_r) + self.B_r
        )

        # Current memory content:
        h_hat = np.tanh(
          np.dot(X,self.W_h) + np.dot(np.multiply(reset_gate, h_prev), self.U_h) + self.B_h
        )

        # Hidden state:
        hidden_state = np.multiply(update_gate,h_prev) + np.multiply((1 - update_gate), h_hat)


        # Classifiers (Softmax) :
        dense = np.dot(hidden_state, self.W) + self.B
        probs = activations.softmax(dense, axis = -1)

        return update_gate, reset_gate, h_hat, hidden_state, dense, probs

        

    def forward(self, X, h_prev):
        x_s,z_s,r_s,h_hat = {}, {},{}, {}
        h_s = {}
        y_s,p_s = {}, {}        

        h_s[-1] = np.copy(h_prev)
        

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]

            z_s[t], r_s[t], h_hat[t], h_s[t], y_s[t], p_s[t] = self.cell_forward(
              x_s[t], 
              h_s[t-1]
            )

        return x_s, z_s, r_s, h_hat, h_s, y_s, p_s
    
    def BPTT(self, outs, Y):

        x_s,z_s, r_s, h_hat, h_s, y_s, p_s = outs

        dW_z, dW_r,dW_h, dW = np.zeros_like(self.W_z), np.zeros_like(self.W_r), np.zeros_like(self.W_h),np.zeros_like(self.W)

        dU_z, dU_r,dU_h, = np.zeros_like(self.U_z), np.zeros_like(self.U_r), np.zeros_like(self.U_h)


        dB_z, dB_r,dB_h,dB = np.zeros_like(self.B_z), np.zeros_like(self.B_r),np.zeros_like(self.B_h),np.zeros_like(self.B)

        dh_next = np.zeros_like(h_s[0]) 
        

        # w.r.t. softmax input
        ddense = np.copy(p_s[self.seq_len - 1])
        ddense[np.arange(len(Y)), np.argmax(Y, axis=1)] -= 1
        #ddense[np.argmax(Y,1)] -=1
        #ddense = y_s[149] - Y
        # Softmax classifier's :
        dW = np.dot(h_s[self.seq_len - 1].T, ddense)
        dB = np.sum(ddense, axis = 0, keepdims = True)

        # Backprop through time:
        for t in reversed(range(1,self.seq_len)):           
                        
            # w.r.t. softmax input
            #ddense = np.copy(probs)
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
            #ddense[np.arange(len(Y)),np.argmax(Y,1)] -=1
            # Softmax classifier's :
            #dW += np.dot(hidden_state.T,ddense)
            #dB += np.sum(ddense,axis = 0, keepdims = True)


            # Curernt memort state :
            dh = np.dot(ddense, self.W.T) + dh_next            
            dh_hat = dh * (1-z_s[t])
            dh_hat = dh_hat * dtanh(h_hat[t])
            dW_h += np.dot(x_s[t].T, dh_hat)
            dU_h += np.dot((r_s[t] * h_s[t-1]).T, dh_hat)
            dB_h += np.sum(dh_hat, axis = 0, keepdims = True)

            # Reset gate:
            dr_1 = np.dot(dh_hat,self.U_h.T)
            dr = dr_1  * h_s[t-1]
            dr = dr * dsigmoid(r_s[t])
            dW_r += np.dot(x_s[t].T,dr)
            dU_r += np.dot(h_s[t-1].T, dr)
            dB_r += np.sum(dr, axis = 0, keepdims = True)

            # Forget gate:
            dz = dh * (h_s[t-1] - h_hat[t])
            dz = dz * dsigmoid(z_s[t])
            dW_z += np.dot(x_s[t].T,dz)
            dU_z += np.dot(h_s[t-1].T,dz)
            dB_z += np.sum(dz, axis = 0, keepdims = True)


            # Nexts:
            dh_next = np.dot(dz, self.U_z.T) + (dh * z_s[t]) + (dr_1 * r_s[t]) + np.dot(dr, self.U_r.T)

        # List of gradients :
        grads = [dW, dB, dW_z, dU_z, dB_z, dW_r, dU_r, dB_r, dW_h, dU_h, dB_h]

        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1], grads
    


    def fit(self, X, Y, X_val, y_val, epochs = 50, optimizer = 'SGD', verbose = True, crossVal = False):
        """
        Given the traning dataset,their labels and number of epochs
        fitting the model, and measure the performance
        by validating training dataset.
        """
                
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(X.shape[0])   

            h_prev = np.zeros((self.batch_size, self.hidden_dim))

            for i in range(round(X.shape[0] / self.batch_size) - 1): 
          
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
               
                # Forward + BPTT + SGD:
                cache_train = self.forward(X_feed,h_prev)
                h,grads = self.BPTT(cache_train, y_feed)

                if optimizer == 'SGD':                                                                
                  self.SGD(grads)

                elif optimizer == 'AdaGrad' :
                  self.AdaGrad(grads)

                elif optimizer == 'RMSprop':
                  self.RMSprop(grads)
                
                elif optimizer == 'VanillaAdam':
                  self.VanillaAdam(grads)
                else:
                  self.Adam(grads)

                # Hidden state -------> Previous hidden state
                h_prev = h

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed, cache_train[6][self.seq_len - 1])
            predictions_train = self.predict(X)
            acc_train = accuracy(
              np.argmax(Y, axis = 1), 
              predictions_train
            )

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0], self.hidden_dim))
            _,__,___,____,_____,______, probs_test = self.forward(X_val,test_prevs)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[self.seq_len - 1])
            predictions_val = np.argmax(probs_test[self.seq_len - 1], 1)
            acc_val = accuracy(
              np.argmax(y_val, axis = 1),
              predictions_val
            )

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)
      
    
    def params(self):
        """ Return all weights/biases in sequential order starting from end in list form. """        
        return [
          self.W, self.B, self.W_z, 
          self.U_z, self.B_z, self.W_r,
          self.U_r, self.B_r,self.W_h, 
          self.U_h, self.B_h
        ]

    def SGD(self,grads):
      """

      Stochastic gradient descent with momentum on mini-batches.
      """
      prevs = []
      for param, grad,prev_update in zip(self.params(),grads,self.previous_updates):            
          delta = self.learning_rate * grad - self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)

      self.previous_updates = prevs       

      self.learning_rate *= 0.99999   

    
    def AdaGrad(self, grads):
      """ AdaGrad adaptive optimization algorithm. """         

      for i, (param, grad) in enumerate(zip(self.params(), grads)):
        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

    def RMSprop(self, grads, decay_rate = 0.9):
      """ RMSprop adaptive optimization algorithm """

      for i, (param, grad) in enumerate(zip(self.params(), grads)):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        


    def VanillaAdam(self, grads, beta1 = 0.9, beta2 = 0.999):
        """ Adam optimizer, but bias correction is not implemented """
      
        for i, (param, grad)  in enumerate(zip(self.params(), grads)):
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)



    def Adam(self, grads, beta1 = 0.9, beta2 = 0.999):
        """ Adam optimizer, bias correction is implemented. """

        for i, (param, grad) in enumerate(zip(self.params(), grads)):
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)

        self.t += 1
    
    
    def CategoricalCrossEntropy(self,labels,preds):
        """
        Computes cross entropy between labels and model's predictions
        """
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N
    
    def predict(self,X):
        """
        Return predictions, (not one hot encoded format)
        """

        # Give zeros to hidden/cell states:
        pasts = np.zeros((X.shape[0], self.hidden_dim))
        _,__,___,____,_____,______, probs = self.forward(X, pasts)
        return np.argmax(probs[self.seq_len - 1], axis=1)

    def history(self):
        return {
          'TrainLoss' : self.train_loss,
          'TrainAcc'  : self.train_acc,
          'TestLoss'  : self.test_loss,
          'TestAcc'   : self.test_acc
        } 