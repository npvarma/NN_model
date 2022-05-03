# NN_model

A simple Neural Network model implemented in Python to learn and apply the theory. Includes regularization methods and mini-batch gradient descent suited for binary classifcation problems. Batch normalization, ADAM optimization and multi-nomial classifications are not implemented yet. Intended use is for learning the basics of Machine Learning rather than production runs where existing libraries are (obviously) more efficient.


Dependencies:
numpy, scipy (light), matplotlib


Usage:

>import NN_model as NN

>NN1 = NN.neuralnet(layers, act_hidden, act_output, init)

>NN1.lam=0.01        #lambda for L2 normalization

>NN1.keep_prob=0.9   #keep probability for drop out


>cost = NN1.train(X_train, Y_train, learning_rate, epochs, batch_size, convergence_criteria,
                 printoutput, plot, frequency)
              
>Ypredict=NN1.predict(X_test)


Description:

- X,Y - numpy.ndarray with same no. of training examples.
- X.shape=[no of features, no of training examples]
- Y.shape=[no. of output parameters, no of training examples]

Layers:
- Layers is a Python list that is of the form:
[number of input features, no of nodes in layer 1, no of nodes in layer 2, ... no of output features]

Example: layers=[30 10 5 5 1] for a classification problem with 30 input features.3 hidden layerswith 10,5 and 5 nodes respectively. 


Most of the below parameters have defaults set, so it is not necessary to input them explicitly. 

Activation functions: 
- act_hidden: str, default:relu
- act_output: str, defualt:sigmoid
- Options: relu, sigmoid, tanh, leakyrelu

Initilization methods:
- init: str
- random (natural distribution), xavier, glorot

Training Hyperparameters (to be expanded in future):
- learning_rate: float
- number of epochs: int
- conv_cost (cost function for convergence): float
- printcost: boolean
- plot: boolean
- report_freq: int
- regularization - dropout and L2 controlled via neuralnet.lam and neuralnet.keep_probability.
- batch size: int

Misc:

To save a trained network with its parameters:
> NN.save_nn(network, filename)

To load a previously saved network:
> network=NN.load_nn(filename)
