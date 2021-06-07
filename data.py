import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import os

os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz')
os.system('tar -zxvf MNIST.tar.gz')

def mnist():
    
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    
    # Download and load the train data
    trainset = datasets.MNIST(root = './', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,drop_last=True)
    
    # Download and load the test data
    testset = datasets.MNIST(root = './', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True,drop_last=True)
    
    # Download and load the test data
    #testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainloader, testloader
