import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
from tensorflow import keras


#Fully Connected Neural Network class for its initialization and further methods like the training and test. Time computation is quite surprisingly due to the matrix operations with relative smallamount of datasets compared to life example datasets for training (near 10'000).
class NeuralNetwork:
    
    #Function for initializing  the Weights ands Biases of each layer of the NN accoirding to the specified architecture.
    #Inputs:
        #- input_size, number of hidden layers, number of neurons (list of numbers of neurons per each hidden layer), number of nodes for output, number of iterations, learning rate, activation function to use, penalty value (default is 0.0), parameter to define if softmax is to be used at the end or not (not recomenen for regression problems).
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
    
    #Softmax function used in the last layer for targeting probabilities
    #Inputs:
		#- value x.
	#Output:
		#- function evaluated in that value x.
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    
    #Function for evaluationg which activation function to use according to the desired activation function initialized in the Neural Network
    #Inputs:
		#- value x.
	#Output:
		#- function evaluated in that value x..
    def activation(self, x):
        if self.act_type == 'relu':
            return self.ReLu(x)
        elif self.act_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.act_type == 'tanh':
            return self.tanh(x)
        elif self.act_type == 'leaky_relu':
            return self.Leaky_ReLu(x)
    
    #Function for evaluationg which derivate function to use according to the desired activation function initialized in the Neural Network
    #Inputs:
		#-  value x.
	#Output:
		#- function evaluated in that value x.
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
        #- Initial parameters of weights, data set, biases and probability boolean to decide if Aoftmax is used or not.
    #Output:
        #- Calculated A and Z values for each hidden layer.
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
        #- Initial parameters of weights, data set, biases, A and Z values of the hidden layers.
    #Output:
        #- Gradients for Weights and Biases in each hidden layer to use in the upgrade step.
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
        #- Weights and biases, gradients for update, learning rate and penalty parameter (0.0 if there is no penalty).
    #Output:
        #- Updates Weights and Biases per hidden layer.
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
        #-
    #Output:
        #-
    def train(self):
        for i in range(self.epochs):
            #print(i)
            self.Z, self.A = self.feed_forward(self.X_input, self.W, self.B, self.prob)
            self.dW, self.dB = self.back_propagation(self.X_input, self.Y, self.W, self.B, self.A, self.Z)
            self.W, self.B = self.upgrade_parameters(self.dW, self.dB, self.W, self.B, self.lr, self.penalty)
            
    #Predict function.
    #Based on an already train NN, it predicts the classes or output of a test set passed as argument.
    #Input:
        #- Test_set in the same type as the Train set used to initialize the NN.
    #Output:
        #- Values predicted or probabilities per nodes. To use as it is for regression problems, or probability logits to be decoded with the decoder function in method.py    
    def predict(self, test_set):
        Zetas, As = self.feed_forward(test_set, self.W, self.B, self.prob)
        classes = As['A'+str(len(self.num_nodes)+1)]
        return classes


#Logistic Regression class and further methods like the training and test.
class Logistic_Regression:
    
    #Function for initializing the Parameters, including the initial coefficients from 0 to 1.
    #Inputs:
        #- input data, target values, number of iterations, learning rate, penalty value (default is 0.0), threshold for binary classification in probability distribution (default is 0.5).
    #Output:
        #- initialized values.
    def __init__(self, X_input, Y_input, epochs, lr, penalty=0.0, threshold=0.5):
        self.X = X_input
        self.n_inputs, self.n_features = X_input.shape
        self.Y = Y_input
        self.lr = lr
        self.B = np.random.rand(self.n_features,1)
        self.penalty = penalty
        self.epochs = int(epochs)
        self.prob = threshold
    
    #Probability calculation function (Sigmoid function)
    #Inputs:
        #- values (array of values in column).
    #Output:
        #- evaluated values in sigmoid fucntion (array with size equal to the Input).
    def probability(self, values):
        return 1/(1 + np.exp(-values))
    #Train function.
    #Do all the processes of gradient descent, with a cost function defined on probabilty comparison. Penalty parametr also taked into accountto compute another gradient regularized in case that penalty is different from 0.0
    #Input:
        #-
    #Output:
        #-
    def train(self):
        t0, t1 = 5, 50
        #print(self.B)
        for i in range(self.epochs):
            if self.penalty != 0.0:
                G = self.X.T @ (self.Y - self.probability( self.X @ self.B )) + (self.penalty * self.B)
            else:
                G = self.X.T @ (self.Y - self.probability( self.X @ self.B ))
            
            self.B += self.lr * G
    
    #Predict function.
    #Based on an already train Logistic Regression (updated coefficients).
    #Input:
        #- Test_set in the same type as the Train set used to initialize the class.
    #Output:
        #- Values predicted in the way of probabilities. It instantly translates to the desired class (0 or 1) (binary classification).   
    def predict(self, values):
        
        results = self.probability( values @ self.B )
        results[results < self.prob] = 0
        results[results >= self.prob] = 1
        return results
    
    
#U-Net Architecture for image segmentation using Keras. Only works with square images
class U_Net:
    
    #Function for initializing  the Weights ands Biases of each layer of the NN accoirding to the specified architecture.
    #Inputs:
        #- input_size, number of hidden layers, number of neurons (list of numbers of neurons per each hidden layer), number of nodes for output, number of iterations, learning rate, activation function to use, penalty value (default is 0.0), parameter to define if softmax is to be used at the end or not (not recomenen for regression problems).
    #Output:
        #- the entire architecture with initialized weights and biases (type dictionary).
    def __init__(self, X_input=None, Y_input=None, batch_size=None, learning_rate=None, epochs=None, act_type='relu', out_act_type='sigmoid', penalty=0.0):
        
        self.X_input = X_input
        self.Y_input = Y_input
        
        self.num_samples, self.img_height, self.img_width, self.img_chanels = X_input.shape
        #self.img_height = X_input.shape[1]
        #self.img_chanels = X_input.shape[2]
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        #self.proba = proba
        self.batch_size = batch_size
        
        self.act_type = act_type
        
        if self.act_type == 'leaky_relu':
            self.act_type =  keras.layers.LeakyReLU(alpha=0.1)
            
        self.out_act_type = out_act_type
        
        self.inputs = keras.layers.Input((self.img_height, self.img_width, self.img_chanels))
        
        #self.normal_inp = keras.layers.Lambda(lambda x: x / 255)(self.inputs)
        
        self.normal_inp = self.inputs
        
        feat5, feat4, feat3, feat2, feat1 = int(self.img_height*2), int(self.img_height), int(self.img_height/2), int(self.img_height/4), int(self.img_height/8)
        
        #Down of the U-Net Part
        
        #First layer of the U-Net
        self.conv1 = keras.layers.Conv2D(feat1, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.normal_inp)
        self.conv1 = keras.layers.Dropout(0.1)(self.conv1)
        self.conv1 = keras.layers.Conv2D(feat1, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.conv1)
        self.pool1 = keras.layers.MaxPooling2D((2,2))(self.conv1)
        
        #Second Layer of the U-Net
        self.conv2 = keras.layers.Conv2D(feat2, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.pool1)
        self.conv2 = keras.layers.Dropout(0.1)(self.conv2)
        self.conv2 = keras.layers.Conv2D(feat2, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.conv2)
        self.pool2 = keras.layers.MaxPooling2D((2,2))(self.conv2)
        
        #Third Layer of the U-Net
        self.conv3 = keras.layers.Conv2D(feat3, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.pool2)
        self.conv3 = keras.layers.Dropout(0.2)(self.conv3)
        self.conv3 = keras.layers.Conv2D(feat3, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.conv3)
        self.pool3 = keras.layers.MaxPooling2D((2,2))(self.conv3)
        
        #Fourth Layer of the U-Net
        self.conv4 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.pool3)
        self.conv4 = keras.layers.Dropout(0.2)(self.conv4)
        self.conv4 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.conv4)
        self.pool4 = keras.layers.MaxPooling2D((2,2))(self.conv4)
        
        #Fifth and Bottom Layer of the U-Net
        self.conv5 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.pool4)
        self.conv5 = keras.layers.Dropout(0.3)(self.conv5)
        self.conv5 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.conv5)
        
        
        #Up fo the U-Net Part and start concatenatting (By this we realized we could have done all thise shortly with a Sequential() function and a loop, but okay, seems more understandable and give us more control).
        
        #Sixth layer
        self.up4 = keras.layers.Conv2DTranspose(feat4, (2,2), strides=(2,2), padding='same')(self.conv5)
        self.up4 = keras.layers.concatenate([self.up4, self.conv4])
        self.upconv4 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.up4)
        self.upconv4 = keras.layers.Dropout(0.2)(self.upconv4)
        self.upconv4 = keras.layers.Conv2D(feat4, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.upconv4)
        
        #Seventh layer
        self.up3 = keras.layers.Conv2DTranspose(feat3, (2,2), strides=(2,2), padding='same')(self.upconv4)
        self.up3 = keras.layers.concatenate([self.up3, self.conv3])
        self.upconv3 = keras.layers.Conv2D(feat3, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.up3)
        self.upconv3 = keras.layers.Dropout(0.2)(self.upconv3)
        self.upconv3 = keras.layers.Conv2D(feat3, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.upconv3)
        
        #Seventh layer
        self.up2 = keras.layers.Conv2DTranspose(feat2, (2,2), strides=(2,2), padding='same')(self.upconv3)
        self.up2 = keras.layers.concatenate([self.up2, self.conv2])
        self.upconv2 = keras.layers.Conv2D(feat2, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.up2)
        self.upconv2 = keras.layers.Dropout(0.1)(self.upconv2)
        self.upconv2 = keras.layers.Conv2D(feat2, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.upconv2)
        
        #Eighth layer
        self.up1 = keras.layers.Conv2DTranspose(feat1, (2,2), strides=(2,2), padding='same')(self.upconv2)
        self.up1 = keras.layers.concatenate([self.up1, self.conv1], axis=3)
        self.upconv1 = keras.layers.Conv2D(feat1, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.up1)
        self.upconv1 = keras.layers.Dropout(0.1)(self.upconv1)
        self.upconv1 = keras.layers.Conv2D(feat1, (3,3), activation=self.act_type, kernel_initializer='he_normal', padding='same')(self.upconv1)
        
        
        self.output = keras.layers.Conv2D(1, (1,1), activation=self.out_act_type)(self.upconv1)
        
        self.model = keras.Model(inputs=[self.inputs], outputs=[self.output])
        
        #the model that contains the backpropagation algorithms. We define here two different optimizers.
        if self.learning_rate is not None:
            sgd = keras.optimizers.SGD(lr=self.learning_rate)
        else:
            sgd = 'adam'
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        
    def summary(self):
        return self.model.summary()
    
    def train(self, verbose_opt):
        #We are going to mintor by setting a threshold that if the iterations score are not changing, meaning that the model could have reach  the optimu, then it stopps the iterations.
        call = keras.callbacks.EarlyStopping(patience=15, monitor='accuracy') 
        return self.model.fit(self.X_input, self.Y_input, epochs=self.epochs, validation_split=0.1, batch_size=self.batch_size, shuffle=True, callbacks=[call], verbose=verbose_opt)
        
    #We do not want to miss a possible good model because we can need this in more than just this project3, so we are gopnna save the model in the h5 file for later use outside the class.
    def checkpoint(self):
        self.checkpoint = keras.callbacks.ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
        
    #makes the prediciton once the model is trained. WARNING: image dataset to predict must be in the same shape (height x width) as the dataset used for training.
    def predict(self, X_predict):
        return self.model.predict(X_predict) #verbose=1