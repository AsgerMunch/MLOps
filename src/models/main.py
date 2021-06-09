import sys
import argparse
import torch

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import nn, optim
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, '/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/src/data')
from make_dataset import main
from model import MyAwesomeModel

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
hp_params = {'nfilters': 16, 'kernel': 4, 'stride': 2, 'padding': 2, 'nfeatures': 100}

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
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        trainloader, _ = main()
        
        # Train the network
        train_loss = []
        epochs = 2
        for e in range(epochs):
            print("Epoch {}".format(e+1))
            running_loss = 0
            count = 0
            for images, labels in trainloader:
                if count == 1:
                    grid = torchvision.utils.make_grid(images)
                    writer.add_image('images', grid, 0)
                    writer.add_graph(model,images)
                # Flatten images
                #images = images.view(images.shape[0],-1)
                #print("Training number: {}".format(count))
                #print("Input shape: {}".format(images.shape))

                # Training pass
                output = model(images)
                loss = criterion(output,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update total loss
                running_loss += loss.item()
                writer.add_scalar('Train loss', loss.item(), count)
                count+=1
            else:
                train_loss.append(running_loss/len(trainloader))
        
        final_loss = {'loss': loss.item()}
        writer.add_hparams(hp_params,final_loss)
        print('Finished training!')
        writer.close()
        print('Closed the writer off :)')
        checkpoint = {'state_dict': model.state_dict()}
        torch.save(checkpoint, 'checkpoint.pth')
        print("Stored the state!")
        
        plt.plot(np.arange(epochs)+1,train_loss)
        plt.savefig('/Users/asgermunch/Documents/DTU/MLOps/Day 2/Organised code/reports/figures/training curve')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
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
        
        ## Compute test accuracy
        acc_mean = []
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                acc_mean.append(accuracy.item())
        print(f'Accuracy: {sum(acc_mean)/len(testloader)*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    