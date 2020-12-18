import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math as m
import os
import random
import seaborn as sns
import rasterio as rt
from rasterio.plot import show as rtshow
from rasterio.merge import merge
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
    values = values.astype(int)
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
    plt.tight_layout()
    if fig_name != None:
        plt.savefig('figures/'+fig_name)

#Central function for loading the images of the data directory and making pre-processing with data augmentation
#Inputs:
    #- name of the directories inside the data folder.
    #- pixel number to devide the main images in square mini-images (batches).
    #- number for stacking the bands into a single image (default is 0 but mainly 2 is used because is the size of bands per pixel).
#Output:
    #- The complete dataset to use and their respective binary image solution.       
def data_load(dirs, pix_division, stack_axis=0):
    
    Y = []
    X = []
    for subfolder in dirs:
        path_y = r'data/'+subfolder+'/glimsRast.tif'
        path_x = r'data/'+subfolder
        y_item = rt.open(path_y).read(1)
        y_item[np.isnan(y_item)] = 0.0
        y_item[y_item != 0.0] = 1.0
        x_item = pre_processing(path_x, stack_axis=stack_axis)
        #print(type(y_item), type(x_item), y_item.shape, x_item.shape)
        
        x_item, y_item = data_augmentation(x_item, y_item, pix_division)
        
        Y += y_item
        X += x_item
        
    return np.array(X), np.array(Y)
        
        

#Pre-processing function. Compute the bands to use in a sort of RGB normal image (samep position).
#Inputs:
    #- name of the individual directory to read.
    #- number for stacking the bands into a single image (default is 0 but mainly 2 is used because is the size of bands per pixel).
#Output:
    #- The image 3-band with each one being a computed band from the original bands available in the sub directory data folder. 
def pre_processing(directory, stack_axis=0):
    path = [directory+'/'+d for d in os.listdir(directory) if not d.endswith('Rast.tif') and d.endswith('.tif')]
    path = np.sort(path)
    layers = []
    for file in path:
        band = rt.open(file).read(1)
        band[np.isnan(band)] = 0.000000001
        layers.append(band)
    #stack them as height, width, channels shape
    im = np.stack(layers, axis=stack_axis)
    
    #Blue band and normalizing above 0
    B1 = im[:,:,0]
    B1 -= B1.min()
    B1 /= B1.max() 
    #calculating snow index and normalizing above 0
    #B2 = (im[:,:,1] - im[:,:,4]) / (im[:,:,1] + im[:,:,4])
    B2 = im[:,:,4] / im[:,:,3]
    B2 -= B2.min()
    B2 /= B2.max()
    #calculating water index and normalizing above cero
    #B3 = 0.2125*im[:,:,5] + 0.7154*im[:,:,1] + 0.0721*im[:,:,0]
    B3 = (im[:,:,1] - im[:,:,2]) / (im[:,:,1] + im[:,:,2])
    #B3 = im[:,:,4]
    B3 -= B3.min()
    B3 /= B3.max()
    
    im = np.stack([B3, B1, B2], axis=stack_axis)
    
    return im

#Data agumentation functino to create more synthetical data from reflection and rotation of each by 90 degrees.
#Inputs:
    #- name of the sub directory inside the data folder.
    #- pixel number to devide the main images in square mini-images (batches).
    #- solution image to do the same data augmentation process so all the synthetic dataset produced matches with also affected solutions based on the same procedures.
#Output:
    #- A collection fo new and more datasets, icnluding the original mini-images and their synthetic produced ones based ont heir originals.  
def data_augmentation(data, target, pix_num):
    #Remember, data is multiband, and target is only a matrix
    #cut_data
    data = np.array(data)
    target = np.array(target)
    new_data = []
    new_targets = []
    height, width, channels = data.shape
    h_loop = int(height/pix_num)
    w_loop = int(width/pix_num)
    for i in range(h_loop):
        for j in range(w_loop):
            mini_im = data[pix_num*i:pix_num*(i+1),pix_num*j:pix_num*(j+1),:]
            new_data.append( mini_im )
            
            t_data = np.moveaxis(mini_im, [0,1,2], [1,0,2])
            new_data.append( t_data )
            
            mini_tag = target[pix_num*i:pix_num*(i+1),pix_num*j:pix_num*(j+1)]
            new_targets.append( mini_tag )
            
            t_tag = mini_tag.T
            new_targets.append( t_tag )
            
            #rotation
            for k in range(3):
                mini_im = np.rot90(mini_im)
                new_data.append( mini_im )
                
                t_data = np.rot90(t_data)
                new_data.append( t_data )
                
                mini_tag = np.rot90(mini_tag)
                new_targets.append( mini_tag )
                
                t_tag = np.rot90(t_tag)
                new_targets.append( t_tag )
    
    return new_data, new_targets

#Splitting the image function to create a collection of several images of the same one (similar to inside functions of the data augmentation).
#Inputs:
    #- individual image to segment, normally made for the targets or predicted outputs.
#Output:
    #- an array that contaisn the segmented parts of the original image.
    #- dimensions of the original image before segmentation so data_unit can make the reverse process.
def data_split(data, pix_num):
    #Remember, data is multiband, and target is only a matrix
    #cut_data
    new_data = []
    height, width, channels = data.shape
    h_loop = int(height/pix_num)
    w_loop = int(width/pix_num)
    for i in range(h_loop):
        for j in range(w_loop):
            mini_im = data[pix_num*i:pix_num*(i+1),pix_num*j:pix_num*(j+1),:]
            new_data.append( mini_im )
    
    return np.array(new_data), [height, width, channels]

#A contrary function to the data_split in order to unify the pieces given from data_split and to reconstruct the original image.
#Inputs:
    #- array of segmented images that are part of the orignal.
    #- the original shape of the segmented image before.
#Output:
    #- the original image reconstructed.
def data_unit(data, original_size):
    #Remember, data is multiband, and target is only a matrix
    #cut_data
    new_data = []
    samples, height_now, width_now, channels_now = data.shape
    height, width, channels = original_size
    new_data = np.zeros((height, width, channels))
    h_loop = int(height/height_now)
    w_loop = int(width/width_now)
    count = 0
    for i in range(h_loop):
        for j in range(w_loop):
            new_data[height_now*i:height_now*(i+1),width_now*j:width_now*(j+1)] = data[count]
            count += 1
    
    return np.array(new_data)

#functino for shuffling the dataset several times and splitting it into Train and Test datasets based on a given ratio.
#Inputs:
    #- Input datasets after all the pre-processing.
    #- Solution or target corresponding to the input datasets after all the pre-processing.
    #- Shuffle desition (default is True).
    #- Ratio of the dataset to be used as test.
    #- The type model that this data is prepared for. If "unet", then it gives all in terms of mini-images. If "nn", then gives it all reshaped as list of the pixel values in each band.
#Output:
    #- The dataset (for train and test sets) splitted (both the inputs and targets/outputs/solutions).
def train_test_split(X, Y, shuffle=True, test_perc = 0.2, type_is='unet'):
    print('Preparing Train and Test splitting...')
    #returns X_train, Y_train, X_test, Y_test
    if type_is == 'unet':
        if shuffle == True:
            num_im = list(range(X.shape[0]))
            #shuffle two times ;)
            random.shuffle(num_im)
            random.shuffle(num_im)
            X, Y = X[num_im], Y[num_im]
        
    if type_is == 'nn':
        
        X, Y = X.reshape((X.shape[0] * (X.shape[1] * X.shape[2]), X.shape[3])), Y.reshape((Y.shape[0] * (Y.shape[1] * Y.shape[2]),1))
        
        if shuffle == True:
            num_im = list(range(X.shape[0]))
            #shuffle two times ;)
            random.shuffle(num_im)
            X, Y = X[num_im], Y[num_im]
        
    X_train, Y_train, X_test, Y_test = X[int(X.shape[0]*test_perc):], Y[int(X.shape[0]*test_perc):], X[:int(X.shape[0]*test_perc)], Y[:int(X.shape[0]*test_perc)]
    print('Train and Test splitting done. Total trian set: '+str(X_train.shape[0])+', Total test set: '+str(X_test.shape[0])+'.')
    
    return X_train, Y_train, X_test, Y_test
    