#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:40:28 2020

@author: georgio
"""

import sys

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

import functionsGANs
import classesGANs

from utils import Logger


##LOAD DATA
## data come from http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz  || 60k images of hand-written digits
DATA_FOLDER = '/Users/georgio/MyStuff/MyData/MNIST'
# Load data
data = functionsGANs.mnist_data(DATA_FOLDER)
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# Neural Nets
discriminator = classesGANs.DiscriminatorNet()
generator = classesGANs.GeneratorNet()

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()


def train_discriminator( optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, functionsGANs.ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, functionsGANs.zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake



def train_generator( optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, functionsGANs.ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error



# Number of steps to apply to the discriminator relative to number of steps of generator (GT: i think)
d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1

# Number of epochs
num_epochs = 200
#num_epochs = 1

num_test_samples = 16
test_noise = functionsGANs.noise(num_test_samples)




# Create logger instance
logger = Logger(model_name='ffGAN', data_name='MNIST')
# Total number of epochs to train
# num_epochs = 200
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(functionsGANs.images_to_vectors(real_batch))
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(functionsGANs.noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(functionsGANs.noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = functionsGANs.vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
            # Display status Logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)
            
            
            
            






