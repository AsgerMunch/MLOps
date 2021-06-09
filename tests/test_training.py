#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:24:28 2021

@author: asgermunch
"""

import pytest
import torch
from torch import nn, optim
import numpy as np
import sys
sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/data')
from make_dataset import main
sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/models')
from model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainloader, _ = main()
    
    images, labels = next(iter(trainloader))
    output = model(images)
    loss = criterion(output,labels)
    # Ensure loss is not nan
    assert loss.item() == loss.item()
    
    print("Test sucessful!")

@pytest.mark.parametrize("lr", [0.1, 1, 10])
def test_model_lr(lr):
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    trainloader, _ = main()
    
    images, labels = next(iter(trainloader))
    output = model(images)
    loss = criterion(output,labels)
    # Ensure loss is not nan
    assert loss.item() == loss.item()
    
    print("Test sucessful!")
