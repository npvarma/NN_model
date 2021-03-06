#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:55:38 2022

@author: nvarma
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
from scipy.special import expit
import pickle
import time


matplotlib.interactive(False)
old_settings = np.seterr(all='warn', divide='raise', under='raise')
errmax = 1e6
errmin = 1e-10


class neuralnet:
    def __init__(self, layers, act_hidden='relu', act_output='sigmoid', init='xavier', lam=0, keep_prob=1):
        self.layers = layers
        self.act_hidden = act_hidden
        self.act_output = act_output
        self.parameters = {}
        self.grads = {}
        # bad idea self.globalcache=[]
        self.init = init
        self.lam = lam
        self.kp = keep_prob
        self.dp = 1*(keep_prob < 1)

    def init_params(self):
        method = self.init
        layers = self.layers
        # parameters = {}
        L = len(layers)

        if method == None or method == 'random':
            for l in range(1, L):
                self.parameters['W'+str(l)] = np.random.randn(layers[l],
                                                              layers[l-1])
                self.parameters['b'+str(l)] = np.zeros((layers[l], 1))
        elif method == 'xavier':
            for l in range(1, L):
                self.parameters['W'+str(l)] = np.random.randn(layers[l], layers[l-1])\
                    * np.sqrt(2/layers[l])
                self.parameters['b'+str(l)] = np.zeros((layers[l], 1))
        elif method == 'glorot':
            for l in range(1, L):
                self.parameters['W'+str(l)] = np.random.randn(layers[l], layers[l-1])\
                    * np.sqrt(2/(layers[l-1]+layers[l]))
                self.parameters['b'+str(l)] = np.zeros((layers[l], 1))
        else:
            raise ValueError
        print("Initialization complete")

    def activation(self, Z, act):
        if act == 'sigmoid':
            A = expit(Z)
        elif act == 'relu':
            A = np.maximum(np.zeros(Z.shape), Z)
        elif act == 'tanh':
            A = np.tanh(Z)
        elif act == 'leakyrelu':
            A = np.maximum(0.1*Z, Z)
        else:
            raise(ValueError('Invalid activation function'))
        return A

    def dz(self, dA, Aprev, W, b, act):
        # A, Z , W = cache
        Z = np.dot(W, Aprev)+b
        if act == 'relu':
            dg = (Z > 0)*1
        elif act == 'sigmoid':
            dg = expit(Z)*(1-expit(Z))
        elif act == 'tanh':
            dg = 1-np.power(np.tanh(Z), 2)
        elif act == 'leakyrelu':
            dg = (Z > 0)*1+(Z <= 0)*-0.1

        # print(act)
        # print(Z.shape)
        # print(dg.shape)
        # print(dA.shape)
        dZ = np.multiply(dA, dg)

        return dZ

    def fwd_layer(self, Aprev, W, b, act, l, predict):
        Z = np.dot(W, Aprev)+b
        A = self.activation(Z, act)
        D = (self.dp*np.random.uniform(0, 1, A.shape) < self.kp)

        if (predict == False and l < (len(self.layers)-1) and l > 0):
            A = np.multiply(A, D)/self.kp

        cache = (Aprev, W, b, D)

        return A, cache

    def fwd_model(self, X, parameters, predict=False):
        layers = self.layers
        L = len(layers)
        A = X  # first layer
        globalcache = []
        for l in range(1, L-1):
            W = parameters['W'+str(l)]
            b = parameters['b'+str(l)]
            A_old = copy.deepcopy(A)
            A, cache = self.fwd_layer(A_old, W, b, self.act_hidden, l, predict)
            globalcache.append(cache)

        l = L-1

        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        A_old = copy.deepcopy(A)
        AL, cache = self.fwd_layer(A_old, W, b, self.act_output, l, predict)
        globalcache.append(cache)
        # print("forward complete")
        return AL, globalcache

    def cost_func(self, Yh, Y):
        m = Yh.shape[1]
        Y.reshape(Yh.shape)
        cost = -1/m*np.sum(np.dot(Y, np.log(np.clip(Yh.T, errmin, errmax)))+np.dot((1-Y),
                                                                                   np.log(np.clip(1-Yh, errmin, errmax).T)), axis=1)
        Warray = self.getWarray()
        cost = cost+self.lam/2/m*np.linalg.norm(Warray)

        return np.squeeze(cost)

    def backprop_layer(self, dA, Dprev, cache, act, l):
        Aprev, W, b, D = cache
        m = Aprev.shape[1]  # no of training examples=n_columns of A
        dZ = self.dz(dA, Aprev, W, b, act)
        dW = 1/m*(np.dot(dZ, Aprev.T)+self.lam*W)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        if (l > 0) and (l < len(self.layers)-2):  # no drop out for input and output layers
            dA_prev = np.multiply(dA_prev, Dprev)/self.kp

        return dA_prev, dW, db

    def backprop_model(self, AL, globalcache, Y):
        L = len(self.layers)
        # A, W, b = self.globalcache
        # Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, np.clip(AL, errmin, errmax)) -
                 np.divide(1-Y, np.clip(1-AL, errmin, errmax)))
        # self.grads["dA"+str(L)] = dAL

        '''
        Warning: Hardcoded
        cost function dependent derivative for the last layer
        dA=-y/yh-(1-y)/(1-yh)
        '''
        # code for last layer begins here
        l = L-2
        current_cache = globalcache[l]
        __, __, __, Dprev = globalcache[l]
        dA_prev_temp, dW_temp, db_temp = self.backprop_layer(dAL, Dprev,
                                                             current_cache, self.act_output, l)
        # self.grads["dA" + str(L-2)] = dA_prev_temp
        self.grads["dW" + str(L-1)] = dW_temp
        self.grads["db" + str(L-1)] = db_temp

        # code for last layer ends

        # code for layers L-1 to 2 - first layer is input.

        for l in range(L-2, 0, -1):
            # print('backprop:'+str(l), len(globalcache))
            current_cache = globalcache[l-1]
            dA_prev = copy.deepcopy(dA_prev_temp)
            dA_prev_temp, dW_temp, db_temp = self.backprop_layer(dA_prev, Dprev,
                                                                 current_cache, self.act_hidden, l)
            # self.grads["dA" + str(l-1)] = dA_prev_temp
            self.grads["dW" + str(l)] = dW_temp
            self.grads["db" + str(l)] = db_temp

        # end back prop

    def update_params(self, learning_rate):
        parameters = copy.copy(self.parameters)
        L = len(self.layers)

        for l in range(L-1):
            # print(l)
            parameters["W" + str(l+1)] = self.parameters['W'+str(l+1)] -\
                learning_rate*self.grads['dW'+str(l+1)]
            parameters["b" + str(l+1)] = self.parameters['b'+str(l+1)] -\
                learning_rate*self.grads['db'+str(l+1)]

        self.parameters = parameters

    def train(self, X, Y, learning_rate=0.01, n_epochs=100, batch_size=64, conv_cost=0.01,
              printcost=True, plot=False, report_freq=500):
        cost = []
        div_warn = 0  # divergence warning
        c = 1
        t1 = time.time()

        # create mini-batch parameters and randomize
        batchnum = np.ceil(X.shape[1]/batch_size).astype('int')
        lastbatch_size = X.shape[1] % batch_size
        m = X.shape[1]
        randomize = list(np.random.permutation(m))
        X_shuffled = X[:, randomize]
        Y_shuffled = Y[:, randomize]

        # if plot:
        #     fig=plt.figure()
        #     axes = fig.add_subplot(111)
        #     axes.set_autoscale_on(True) # enable autoscale
        #     axes.autoscale_view(True,True,True)
        #     l, = plt.plot([], [], 'r-')

        self.init_params()

        for i in range(1, n_epochs+1):

            for j in range(1, batchnum+1):
                if j < batchnum:
                    c1 = (j-1)*batch_size
                    c2 = (j)*batch_size-1
                    X_batch = X_shuffled[:, c1:c2]
                    Y_batch = Y_shuffled[:, c1:c2]
                elif j == batchnum and lastbatch_size > 0:
                    X_batch = X_shuffled[:, -lastbatch_size:]
                    Y_batch = Y_shuffled[:, -lastbatch_size:]

                AL, globalcache = self.fwd_model(X_batch, self.parameters)
                self.backprop_model(AL, globalcache, Y_batch)
                self.update_params(learning_rate)
                cprev = c
                c = self.cost_func(AL, Y_batch)

            if c > cprev:
                div_warn += 1
                # print("Divergence!")
                # return None
                if div_warn > 2*batch_size:
                    print("Stopped due to divergence!")
                    return None
            elif c < cprev:
                div_warn = 0

            if c < conv_cost:
                print("Converged!")
                break

            if i % report_freq == 0:
                t2 = time.time()
                time_epoch = np.round((t2-t1)/report_freq, 3)
                if printcost:
                    print('Epoch:'+str(i)+' Cost: '+str(np.round(c, 3)) +
                          ' Time/epoch: '+str(time_epoch)+'s')
                t1 = time.time()
                if plot:
                    cost.append(c)
                    # plt.close()
                    #plt.plot(range(len(cost)), cost)

        if len(cost) > 0:
            plt.close()
            xe = np.arange(0, n_epochs, n_epochs/(len(cost)))
            plt.plot(xe, cost)
            plt.xlabel("epochs")
            plt.ylabel("cost")
            plt.show()

        return c

    def predict(self, X):
        assert(X.shape[0] == self.layers[0])
        AL, cache = self.fwd_model(X, self.parameters, predict=True)
        # AL = np.around(AL)
        return AL

    def perturb_param(self, e):
        parameters = self.parameters.copy()
        L = len(self.layers)
        for l in range(L-1):
            # print(l)
            parameters["W" + str(l+1)] = self.parameters['W'+str(l+1)] + e
            parameters["b" + str(l+1)] = self.parameters['b'+str(l+1)] + e
        return parameters

    '''def gradientcheck(self, X, Y):
       
        e = 1e-7
        
        dtheta = np.concatenate([x.ravel() for x in grads])
        theta_f = self.perturb_param(e)
        theta_b = self.perturb_param(-e)
        ALf, cache = self.fwd_model(X, theta_f)
        ALb, cache = self.fwd_model(X, theta_b)
        Jf = self.cost_func(ALf, Y)
        Jb = self.cost_func(ALb, Y)
        dtheta_e = (Jf-Jb)/2/e
        dtheta = np.linalg.norm(dtheta, ord=2)
        return dtheta, dtheta_e
        # return np.linalg.norm((dtheta-dtheta_e), ord=2)'''

    def getWarray(self):
        list1 = list(self.parameters.items())
        array1 = np.array(list1, dtype=object)[:, 1]
        array1 = array1[::2]
        warray = np.concatenate([x.ravel() for x in array1])

        return warray

    def getbarray(self):
        list1 = list(self.parameters.items())
        array1 = np.array(list1, dtype=object)[:, 1]
        array1 = array1[1::2]
        barray = np.concatenate([x.ravel() for x in array1])

        return barray


def save_nn(obj, filename):
    file = open('filename'+'.nn', 'wb')
    pickle.dump(obj, file)
    print("Saved")


def load_nn(filename):
    file = open('filename'+'.nn', 'rb')
    pickle.load(file)
    print("Loaded")
    return obj
