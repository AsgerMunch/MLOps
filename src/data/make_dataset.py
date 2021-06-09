# -*- coding: utf-8 -*-
#import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import os




#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz')
    os.system('tar -zxvf MNIST.tar.gz')
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    trainset = datasets.MNIST(root = './', download=True, train=True, transform=transform)
    #trainset = datasets.MNIST('/Users/asgermunch/Documents/DTU/MLOps/Day 1', download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,drop_last=True)
    
    # Download and load the test data
    testset = datasets.MNIST(root = './', download=True, train=False, transform=transform)
    #testset = datasets.MNIST(root = '/Users/asgermunch/Documents/DTU/MLOps/Day 1', download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True,drop_last=True)
    return trainloader, testloader


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
