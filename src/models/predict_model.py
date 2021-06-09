import sys
import argparse
import torch

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import nn, optim
import torch.nn.functional as F

sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/data')

from make_dataset import main
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
    
        
    def predict(self):
        print("Predicting the input!")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        parser.add_argument('--images', default='/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/images.npy')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        images = np.load(args.images)
        # Training loop
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = load_checkpoint(args.load_model_from)
            
        with torch.no_grad():
            model.eval()
            ps = torch.exp(model(torch.tensor(images)))
            top_p, top_class = ps.topk(1, dim=1)
        
        plt.imshow(images[0].squeeze(), cmap='Greys_r');
        print(top_class)
        return top_class

if __name__ == '__main__':
    TrainOREvaluate()
    