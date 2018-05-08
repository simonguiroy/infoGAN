import os
import pdb
import pickle
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

def load_mnist(dataset):
    
    # Loads mnist data from pkl file
    data_dir = os.path.join("data", dataset)
    with open(os.path.join(data_dir, "mnist_data.pkl"), "rb") as f:
        mnist_data = pickle.load(f)

    X = mnist_data['X']
    Y = mnist_data['Y']

    # Reshape to (batch, channels, height, width) and scales in [0., 1.]
    X = X.transpose(0, 3, 1, 2) / 255.

    # Converts X to torch tensor
    X = torch.from_numpy(X).type(torch.FloatTensor)
    Y = torch.from_numpy(Y).type(torch.FloatTensor)
    return X, Y

def load_3Dchairs(transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(os.path.join("data", "3Dchairs"), transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader

def load_synth(transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(os.path.join("data", "synth"), transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader

def print_network(net):
    # Counts the number of parameters
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    # Prints the architecture
    print('\nNETWORK ARCHITECTURE ------------------------')
    print(net)
    print('Total number of parameters : {}k'.format(int(num_params/1e3)))
    print('-----------------------------------------------\n')

def save_loss_plot(train_history, filename, infogan=False):
    plt.figure(figsize=(10,4))

    # Defines the plot
    plt.plot(train_history['D_loss'], color="blue", label='D loss')
    plt.plot(train_history['G_loss'], color="orange", label='G loss')
    if infogan: 
        plt.plot(train_history['info_loss'], color="pink", label='Info loss')
    plt.title('Training Curves', fontweight='bold')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True, color="lightgrey")

    # Saves the figure
    plt.savefig(filename)
    plt.close()

def generate_samples(model, filename):
    model.G.eval()

    # Creates samples and saves them
    plt.figure(figsize=(20,20))
    for i in range(25):

        # Basic z vector
        z = torch.rand((1, model.z_dim))

        # Discrete code
        c_disc = torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[1])).type(torch.FloatTensor)
        for j in range(model.n_disc_code-1):
            c_disc = torch.cat([c_disc, torch.from_numpy(np.random.multinomial(1, model.c_disc_dim * [float(1.0 / model.c_disc_dim)], size=[1])).type(torch.FloatTensor)], dim=1)

        # Continuous code
        c_cont = torch.from_numpy(np.random.uniform(-1, 1, size=(1, model.c_cont_dim))).type(torch.FloatTensor)

        # Converts to Variable (sends to GPU if necessary)
        if model.gpu_mode:
            z = Variable(z.cuda(model.gpu_id), volatile=True)
            c_disc = Variable(c_disc.cuda(model.gpu_id), volatile=True)
            c_cont = Variable(c_cont.cuda(model.gpu_id), volatile=True)
        else:
            z = Variable(z, volatile=True)
            c_disc = Variable(c_disc, volatile=True)
            c_cont = Variable(c_cont, volatile=True)
        
        # Forward propagation
        x = model.G(z, c_cont, c_disc, model.dataset)

        # Reshapes dimensions and convert to ndarray
        x = x.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

        # Adds sample to the figure
        plt.subplot(5,5,i+1)
        plt.imshow(x, cmap='gray')
        plt.axis('off')

    plt.savefig(filename, bbox_inches='tight')
    plt.close()