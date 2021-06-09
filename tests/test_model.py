#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:16:33 2021

@author: asgermunch
"""

import sys
sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/data')
from make_dataset import main
sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/models')
from model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel()
    trainloader, _ = main()
    
    images, labels = next(iter(trainloader))
    output = model(images)
    assert output.shape == (64,10)
    
    print("Test sucessful!")