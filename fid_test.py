#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:40:28 2020

@author: georgio
"""

import sys
import os
import numpy as np


# example of calculating the frechet inception distance
# from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
#import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
 

# calcualte the FID
def calc_fid(a1,a2):
    mu1 = a1.mean(axis=0)
    mu2 = a2.mean(axis=0)
    s1 = cov(a1, rowvar=False)
    s2 = cov(a2, rowvar=False)
    
    #print some shapes
    print('a...shape(mu1)=', np.shape(mu1) )
    print('a...shape(mu2)=', np.shape(mu2) )
    print('a...shape(s1) =', np.shape(s1) )
    print('a...shape(s2) =', np.shape(s2) )
    
    #calcualte the diff of means, squared
    m2diff = np.sum((mu1-mu2)**2)
    #calcualte the product of covs, qrted
    covmean = sqrtm(s1.dot(s2))
    # check if imaginary numbers and send to real
    if iscomplexobj(covmean):
        covmean = covmean.real
    #calculate score
    fid = m2diff + trace( s1 + s2 - 2.0*covmean )
    return fid

# this dim, n2, is the number of images to be averaged and covarianced.
n2=3
# the other dim (n1=10) is the number of nuerons=activations in the layer I am calculating the FID on.
# I can cacl FID on image by flattening it into a vector.
# OR I can calc FID on a deep layer of the DNN so that it picks up the longer wavelength features of image.
# OR I can build a weighted average of FID over all layers in order to 
# have a better representation of fine and coarse features. 

# define two collections of activations
act1 = random(10*n2)
print('a...shape(act1): %.3f' % np.shape(act1) )
act1 = act1.reshape((10,n2))
print('b...shape(act1)=' , np.shape(act1) )
act2 = random(10*n2)
print('a...shape(act2): %.3f' % np.shape(act2) )
act2 = act2.reshape((10,n2))
print('b...shape(act2)=', np.shape(act2) )


mu1 = act1.mean(axis=0)
print('a...shape(mu1) = ', np.shape(mu1) )
mu2 = act2.mean(axis=0)
print('a...shape(mu2) = ', np.shape(mu2) )
s1 = cov(act1, rowvar=False)
print('a...shape(s1) = ', np.shape(s1) )
s2 = cov(act2, rowvar=False)
print('a...shape(s2) = ', np.shape(s2) )
    
# fid between act1 and act1
fid = calc_fid(act1, act1)
print('FID (same): %.3f' % fid)
# fid between act1 and act2
fid = calc_fid(act1, act2)
print('FID (different): %.3f' % fid)


