#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:01:39 2021

@author: asgermunch
"""

import sys

import numpy as np
import torch

from src.data.make_dataset import main

def test_data():
    
    trainloader, testloader = main()
    
    # Check the input size is as expected
    assert len(trainloader) == 937
    assert len(testloader) == 156
    
    # Check the shape is correct
    images, labels = next(iter(trainloader))
    assert images.shape == (64,1,28,28)
    
    # Check all labels are represented
    assert torch.sum(torch.tensor(np.arange(10))==labels.unique()).item() == 10
    
    print("Test sucessful!")

