#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:41:06 2022

@author: nvarma
"""
import numpy as np
from scipy import stats
import NN_model as NN
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# read and preprocess data into X(features,examples), Y(examples,1)
data = datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()
X = data.data
Y = np.reshape(data.target, (-1, 1))
scaler = preprocessing.StandardScaler().fit(X)
# X_scaled=scaler.transform(X)
X_scaled = stats.zscore(X, axis=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T
# end data read

# hyper parameters
iterations = 10000
learning_rate = 0.001
act_hidden = 'leakyrelu'
act_output = 'sigmoid'
init = 'xavier'
frequency = 1000
conv_criteria = 0.001
printoutput = True
plot = False
n_x = X_train.shape[0]
n_y = Y_train.shape[0]
layers = [n_x, 5, 5, 5, n_y]

# end hyper parameters

NN1 = NN.neuralnet(layers, act_hidden, act_output, init)
NN1.lam=0.00
NN1.keep_prob=0.5



#train

cost, acc = NN1.train(X_train, Y_train, learning_rate, iterations, conv_criteria,
                      printoutput, plot, frequency)


# print results
print("Cost at the end of training:"+str(cost))
ypredict = NN1.predict(X_test)
ypredict = np.around(ypredict)
error = Y_test-ypredict
test_accuracy = 100*(1-np.count_nonzero(error)/error.shape[1])
print('Test accuracy is: '+str(test_accuracy))
