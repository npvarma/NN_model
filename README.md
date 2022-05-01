# NN_model
A simple barebones Neural Network model. Under **continuous development** for my own use as I go through the courses.

Usage:

>import NN_model as NN

>NN1 = NN.neuralnet(layers, act_hidden, act_output, init)
>NN1.lam=0.01        #lambda for L2 normalization
>NN1.keep_prob=0.9   #keep probability for drop out


>cost, acc_L2 = NN1.train(X_train, Y_train, learning_rate=0.01, iterations=2500, conv_cost=0.01,
              printcost=True, plot=False, report_freq=500)
              
>Ypredict=NN1.predict(X_test)


Description:


- X,Y - numpy.ndarray with same no. of training examples.
- X.shape=[no of features, no of training examples]
- Y.shape=[no. of output parameters, no of training examples]

Layers:
- Layers is a Python list that is of the form:
[number of input features, no of nodes in layer 1, no of nodes in layer 2, ... no of output features]

Example: layers=[30 10 5 5 1] for a classification problem with 30 input features.3 hidden layerswith 10,5 and 5 nodes respectively. 

Activation functions: 
- act_hidden: str, default:relu
- act_output: str, defualt:sigmoid
- Options: relu, sigmoid, tanh, leakyrelu

Initilization methods:
- init: str
- random (natural distribution), xavier, glorot

Training Hyperparameters (to be expanded in future):
- learning_rate: float
- iterations: int
- conv_cost (cost function for convergence): float
- printcost: boolean
- plot: boolean
- report_freq: int
- regularization - dropout and L2

Misc:

To save a trained network with its parameters:
> NN.save_nn(network, filename)

To load a previously saved network:
> network=NN.load_nn(filename)
