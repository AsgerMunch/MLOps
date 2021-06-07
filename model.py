import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

# Define batch size, channels, width and height
batch = 64
channels = 1
width = 28
height = 28

# Define CNN parameters
num_filters_conv1 = 16
kernel_size_conv1 = 5 # [height, width]
stride_conv1 = 1*2 # [stride_height, stride_width]
num_l1 = 100
padding_conv1 = 2

# Compute right height or width for output of CNN
def compute_conv_dim(dim_size):
    return int((dim_size - kernel_size_conv1 + 2 * padding_conv1) / stride_conv1 + 1)

# Define CNN model
class MyCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        
        # First convolutional layer
        self.conv_1 = Conv2d(in_channels=channels,
                             out_channels=num_filters_conv1,
                             kernel_size=kernel_size_conv1,
                             padding=padding_conv1,
                             stride=stride_conv1)
        
        self.conv_out_height = compute_conv_dim(height)
        self.conv_out_width = compute_conv_dim(width)
        
        # add dropout to network
        self.dropout = Dropout2d(p=0.5)
        # Define input features
        self.l1_in_features = num_filters_conv1 * self.conv_out_height * self.conv_out_width        
        # Add batchnorm
        self.batchnormInput = torch.nn.BatchNorm1d(num_l1)
        
        # Define linear layer
        self.l_1 = Linear(in_features=self.l1_in_features, 
                          out_features=num_l1,
                          bias=True)
        # Define output layer
        self.l_out = Linear(in_features=num_l1, 
                            out_features=self.num_classes,
                            bias=False)
        
    def forward(self, x):
        x = relu(self.conv_1(x))
        x = x.view(batch, self.l1_in_features)
        x = self.dropout(relu(self.batchnormInput(self.l_1(x))))
        return F.log_softmax(self.l_out(x), dim=1)

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)
        
        # Additional functions
        self.dropout = nn.Dropout(p=.4)
        
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        # Make forward pass
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x),dim=1)
        return x
    
