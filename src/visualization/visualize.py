import sys
import argparse
import torch

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import nn, optim
import torch.nn.functional as F

from sklearn.manifold import TSNE

sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/data')

from make_dataset import main

sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/models')

from model import MyAwesomeModel

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
        
    def visualise(self):
        print("Visualising the input!")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        #parser.add_argument('--images', default='/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/images.npy')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        # Training loop
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = load_checkpoint(args.load_model_from)
        _, testloader = main()
        
        with torch.no_grad():
            model.eval()
            extracted_features = []
            label = []
            for images, labels in testloader:
                feature = model(images,return_features=True).numpy()
                extracted_features.append(feature)
                label.append(labels.numpy())
        
        label = np.array(label).reshape(64*156)
        X = np.array(extracted_features).reshape((64*156,100)) 
        embedding = TSNE(n_components=2).fit_transform(X)
        plt.scatter(embedding[:,0],embedding[:,1],c=label)

if __name__ == '__main__':
    TrainOREvaluate()
    