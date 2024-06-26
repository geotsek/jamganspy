#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:40:28 2020

@author: georgio
"""

import sys
import os
import numpy as np

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets


def mnist_data(DATA_FOLDER):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


'''
def mnist_data_select_labels(DATA_FOLDER, select_labels):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    data_all = datasets.MNIST(root=out_dir, train=True, transform=compose)

    indices = []
    indices = np.where( data_all.targets == select_labels[0] )
    for i in range(1,len(select_labels)):
        indices = np.hstack( (indices,  np.where( data_all.targets== select_labels[i] )) )

    data_selected = torch.utils.data.Subset(data_all, indices)

    return data_selected
'''
  
def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

"""
def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data         
"""            
            
            






