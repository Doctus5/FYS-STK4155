import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math as m
import random as ra
import seaborn as sns
from matplotlib.colors import LogNorm

def FrankeFunction(x,y):
    #evaluation of each term
    term1 = 0.75*np.exp(-(0.25+(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



def D_matrix(step, deg, noise=True, Z_normalize=False):
    x = np.arange(0, 1, step)
    y = x
    #meshgrid for the Design_matrix computation
    X, Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    #correction doing normalization and adding noise on the Z value and reacomodating into 1D
    if noise == True:
        Z += 0.5*np.random.rand(Z.shape[0],Z.shape[1])
    X = X.reshape(X.shape[0]*X.shape[1],1)
    Y = Y.reshape(Y.shape[0]*Y.shape[1],1)
    Z = Z.reshape(Z.shape[0]*Z.shape[1],1)
    if Z_normalize == True:
        Z = Z / np.sqrt(np.sum(Z**2))
    
    # some shuffle on the datapoints
    indexes = np.array(list(range(len(Z))))
    indexes = np.random.shuffle(indexes)
    Z = Z[indexes][0]
    X = X[indexes][0]
    Y = Y[indexes][0]
    #Not necessary but just in case, here is the formula
    pos = m.factorial(deg+2)/(m.factorial(deg)*m.factorial(2))
    #creation of return matrix with first column of intercepts
    Xm = np.zeros((len(X),1))
    Xm[:,0] = 1.0
    
    #In these loops the pos variable is implicitly named here
    for i in range(1,deg+1):
        for j in range(i+1):
            #Stacking polinomial terms in the columns of the X matrix
            Xm = np.hstack((Xm, ((X**(j)) * (Y**(i-j)))))
    return Xm, Z


def R2(z_data,z_model):
	return 1- np.sum((z_data - z_model)**2.0) / np.sum((z_data - np.mean(z_data))**2.0)

#Mean Square Error
def MSE(z_data, z_model):
    n = np.size(z_model)
    return np.sum(((z_data-z_model)**2.0)) / n

def Accuracy(z_data, z_model):
    return np.sum(z_data == z_model) / len(z_data)

#Learning Rate function, normally use for the dynamic method in the Stochastic Gradient Descent part.
def learning_rate(t, t0, t1):
	return t0/(t+t1)

#Function to encode MINST values into a onehot vector:
def encoder(values):
    n = len(values)
    ind = np.max(values)
    enc = np.zeros((n, ind+1))
    enc[range(n), values[:,0]] = 1
    return enc

#Function to decode 
#def decoder(onehots):
def decoder(values):
    enc = np.argmax(values, axis=1).reshape(len(values),1)
    return enc

#Gradient Descent method. Can be the Stochastic or Standard (depending on the input).
#Inputs:
    #- Initial Betas or coeffs, X design matrix, Y (prediction target) vector, number of epochs, size of minibatch (in case of standard), method (defines the method to use: 'stochastic' or 'standard', learning rate lr: define one in case the method is 'standard').
#Outputs:
    #- Predicted Betas or coeffs.
def SGD(B0, X, Y, epoch, batch_size=None, method=None, lr = None, penalty=0.0):
    if method == 'stochastic':
        m = int(len(X)/batch_size)
        t0, t1 = 5, 50
        for i in range(epoch):
            for j in range(m):
                #ri = np.random.randint(m)
                ri = ra.sample(range(len(X)),batch_size)
                Xi = X[ri]
                Yi = Y[ri]
                if penalty != 0.0:
                    G = Xi.T @ (Xi @ B0 - Yi) + (penalty * B0)
                else:
                    G = Xi.T @ (Xi @ B0 - Yi)
                if lr != None:
                    B0 = B0 - lr*G
                else:
                    B0 = B0 - learning_rate(epoch*m+j, t0, t1)*G
    if method == 'standard':
        m = int(len(X)/batch_size)
        for i in range(epoch):
            for j in range(m):
                init = batch_size * j
                final = init + batch_size
                Xi = X[init:final]
                Yi = Y[init:final]
                G = Xi.T @ (Xi @ B0 - Yi)
                B0 = B0 - lr*G
    return B0

#Function for plotting trade-offs
#Inputs:
    #- MSE error for test and training, values for the axis labels, title of the figure, y axis scale, fig_name (optional variable if we want to save the image).
#Output:
    #- Figure show and Figure save in a file.
def plot_trade(er_v, tr_v, x_values, y_label, x_label, title, yscale=None, xscale='None', fig_name=None):
    degree = len(er_v)
    plt.figure(figsize=(10,6))
    plt.plot(x_values,er_v, label='test', color='orange')
    plt.plot(x_values,tr_v, label = 'train', color='blue')
    if yscale != None:
        plt.yscale(yscale)
    if xscale != None:
        plt.xscale(xscale)
    plt.grid()
    plt.legend()
    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)
    plt.title(title, fontsize=20)
    if fig_name != None:
        plt.savefig('figures/'+fig_name)
        
#Function for plotting heat maps
#Inputs:
    #- MSE error for test and training (matrices), title of the figure, x and y labels on axis, rotation option for x values in axis, heat map logarithmic scale option, fig_name (optional variable if we want to save the image).
#Output:
    #- Figure show and Figure save in a file.        
def plot_heatmap(matrix, Title, X_label, Y_label, x_values, y_values, x_rot=False, log_scale=False, fig_name=None):
    largo, alto = 7, 7
    #plot for test set
    plt.figure(figsize=(largo, alto))
    if log_scale == True:
        sns.heatmap(matrix.T, annot=False, cmap='rocket_r', vmin=matrix.min(), vmax=matrix.max(), norm=LogNorm(vmin=matrix.min(), vmax=matrix.max()), yticklabels=y_values, xticklabels=x_values)
    else:
        sns.heatmap(matrix.T, annot=False, cmap='rocket_r', vmin=matrix.min(), vmax=matrix.max(), yticklabels=y_values, xticklabels=x_values)
    plt.gca().invert_yaxis()
    #plt.yticks(np.arange(len(y_values))+0.5, np.round((y_values),3), rotation='horizontal')
    plt.yticks(rotation='horizontal')
    if x_rot == True:
        plt.xticks(rotation='vertical')
    #plt.xticks(np.arange(len(x_values))+0.5, np.round((x_values),0), rotation='vertical')
    #plt.imshow(mer_MSE, cmap='YlGnBu', interpolation='nearest')
    plt.ylabel(Y_label, fontsize=15)
    plt.xlabel(X_label, fontsize=15)
    plt.title(Title, fontsize=20)
    if fig_name != None:
        plt.savefig('figures/'+fig_name)

        
        
    