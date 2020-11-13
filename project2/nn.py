import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m

#Fully Connected Neural Network class for its initialization and further methods like the training and test. Time computation is quite surprisingly due to the matrix operations with relative smallamount of datasets compared to life example datasets for training (near 10'000).
class NeuralNetwork:
    #Function for initializing  the Weights ands Biases of each layer of the NN accoirding to the specified architecture.
    #Inputs:
        #- input_size, number of hidden layers, number of neurons (list of numbers of neurons per each hidden layer), number of nodes for output.
    #Output:
        #- the entire architecture with initialized weights and biases (type dictionary).
    def __init__(self, X_input, Y_input, num_nodes, num_outputs, epochs, lr, act_type='relu', penalty=0.0, prob=True):
        self.X_input = X_input
        self.n_inputs, self.n_features = X_input.shape
        self.W, self.B = {}, {}
        self.Z, self.A = {}, {}
        self.dW, self.dB = {}, {}
        self.Y = Y_input
        self.num_outputs = num_outputs
        self.num_nodes = num_nodes
        self.lr = lr
        self.act_type = act_type
        self.penalty = penalty
        self.epochs = int(epochs)
        self.prob = prob
        
        for i in range(len(self.num_nodes)+1):
            if i == 0:
                self.W['W'+str(i)] = np.random.rand(self.n_features, self.num_nodes[i])
                self.B['B'+str(i)] = np.zeros(self.num_nodes[i]) + 0.01
            elif i == len(self.num_nodes):
                self.W['W'+str(i)] = np.random.rand(self.num_nodes[i-1], self.num_outputs)
                self.B['B'+str(i)] = np.zeros(self.num_outputs) + 0.01
            else:
                self.W['W'+str(i)] = np.random.rand(self.num_nodes[i-1], self.num_nodes[i])
                self.B['B'+str(i)] = np.zeros(self.num_nodes[i]) + 0.01
                
    def init_check(self):
        print('report of Data, Weights and Biases shapes at Initialization:')
        print(self.X_input.shape)
        for ind in self.W.keys():
            print(ind, self.W[ind].shape)
        for ind in self.B.keys():
            print(ind, self.B[ind].shape)
        print(self.Y.shape)
        
    #Sigmoid function used as an activation function.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
        #Derivative of the Sigmoid function used as an activation function for backprop.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def dev_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))
    
    #Sigmoid function used as an activation function.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def tanh(self, x):
        return np.tanh(x)
    
        #Derivative of the Sigmoid function used as an activation function for backprop.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def dev_tanh(self, x):
        return 1 - self.tanh(x)**2
    
    #ReLU function used as an activation function.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def ReLu(self, x):
        x[x <= 0] = 0
        return x
    
    #Heaviside function (derivative of ReLu) used as an activation function for backprop.
    #Inputs:
		#- value x.
	#Output:
		#- function evaluated in that value x.
    def dev_ReLu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    #Leaky ReLU function used as an activation function.
    #Inputs:
        #- value x.
    #Output:
        #- function evaluated in that value x.
    def Leaky_ReLu(self, x):
        x[x <= 0] *= 0.01
        return x
    
    #Leaky_ReLU derivative function used as an activation function for backprop.
    #Inputs:
		#- value x.
	#Output:
		#- function evaluated in that value x.
    def dev_Leaky_ReLu(self, x):
        x[x <= 0] = 0.01
        x[x > 0] = 1
        return x
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    
    def activation(self, x):
        if self.act_type == 'relu':
            return self.ReLu(x)
        elif self.act_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.act_type == 'tanh':
            return self.tanh(x)
        elif self.act_type == 'leaky_relu':
            return self.Leaky_ReLu(x)
        
    def derivative(self, x):
        if self.act_type == 'relu':
            return self.dev_ReLu(x)
        elif self.act_type == 'sigmoid':
            return self.dev_sigmoid(x)
        elif self.act_type == 'tanh':
            return self.dev_tanh(x)
        elif self.act_type == 'leaky_relu':
            return self.dev_Leaky_ReLu(x)
        
    #Feed Forwards function
    #Input:
        #- Initial parameters of weights, data set (Design MAtrix) and biases.
    #Output:
        #- Calculated probabilities A and Z values.
    def feed_forward(self, X, W, B, prob):
        iterations = len(W)
        Z = {}
        A = {}
        for i in range(iterations):
            if i == 0:
                Z['Z'+str(i+1)] = X @ W['W'+str(i)] + B['B'+str(i)]
                A['A'+str(i+1)] = self.activation(Z['Z'+str(i+1)])
            elif i == (iterations - 1):
                Z['Z'+str(i+1)] = A['A'+str(i)] @ W['W'+str(i)] + B['B'+str(i)]
                if prob == True:
                    A['A'+str(i+1)] = self.softmax(Z['Z'+str(i+1)])
                else:
                    A['A'+str(i+1)] = Z['Z'+str(i+1)]
            else:
                Z['Z'+str(i+1)] = A['A'+str(i)] @ W['W'+str(i)] + B['B'+str(i)]
                A['A'+str(i+1)] = self.activation(Z['Z'+str(i+1)])
        return Z, A
    
    #Back Propagation function
    #Input:
        #- Initial parameters of weights, data set (Design MAtrix) and biases.
    #Output:
        #- Gradients for Weights and Biases in each hidden layer.
    def back_propagation(self, X, Y, W, B, A, Z):
        layers = len(A)
        m = len(X)
        dW = {}
        dB = {}
        for i in range(layers-1,-1,-1):
            if i == layers-1:
                delta = A['A'+str(i+1)] - Y
                dW['dW'+str(i)] = (1/m) * A['A'+str(i)].T @ delta
                dB['dB'+str(i)] = (1/m) * np.sum(delta, axis=0)
            elif i == 0:
                delta = (delta @ W['W'+str(i+1)].T) * self.derivative(Z['Z'+str(i+1)])
                dW['dW'+str(i)] = (1/m) * X.T @ delta
                dB['dB'+str(i)] = (1/m) * np.sum(delta, axis=0)
            else:
                delta = (delta @ W['W'+str(i+1)].T) * self.derivative(Z['Z'+str(i+1)])
                dW['dW'+str(i)] = (1/m) * A['A'+str(i)].T @ delta
                dB['dB'+str(i)] = (1/m) * np.sum(delta, axis=0)
        return dW, dB
    
    #Parameters Upgrade function
    #Input:
        #- Initial parameters of weights and biases, gradients for update, learning rate and penalty parameter (0 if there is no penalty).
    #Output:
        #- Gradients for Weights and Biases in each hidden layer.
    def upgrade_parameters(self, dW, dB, W, B, lr, penalty):
        for i in range(len(dW)):
            if penalty != 0.0:
                dW['dW'+str(i)] += penalty * W['W'+str(i)]
            W['W'+str(i)] -= lr * dW['dW'+str(i)]
            B['B'+str(i)] -= lr * dB['dB'+str(i)]
        return W, B
    
    #Train function.
    #Do all the processes of feed_forward, back_propagation and upgrade_parameters for a certain amount of epochs until Weights and Biases are updated completely for this training set
    #Input:
    #Output:    
    def train(self):
        for i in range(self.epochs):
            #print(i)
            self.Z, self.A = self.feed_forward(self.X_input, self.W, self.B, self.prob)
            self.dW, self.dB = self.back_propagation(self.X_input, self.Y, self.W, self.B, self.A, self.Z)
            self.W, self.B = self.upgrade_parameters(self.dW, self.dB, self.W, self.B, self.lr, self.penalty)
            #print(self.A['A4'])
            
    #Predict function.
    #Based on an already train NN, it predicts the classes or output of a test set passed as argument.
    #Input:
        #- Test_set in the same type as the Train set used to initialize the NN.
    #Output:
        #- If it is a classification, it returns the position of the maximum probability achieved for each input of the test set. If it is a singular-value prediction, it returns the value predicted.    
    def predict(self, test_set):
        Zetas, As = self.feed_forward(test_set, self.W, self.B, self.prob)
        classes = As['A'+str(len(self.num_nodes)+1)]
        return classes



