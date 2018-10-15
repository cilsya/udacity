## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        # # Using NaimishNet architecture.
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 64, 5)
        # self.conv3 = nn.Conv2d(64, 128, 5)
        # self.conv4 = nn.Conv2d(128, 512, 3)
        # self.fc1 = nn.Linear(512*11*11, 1024)
        # self.fc1_drop = nn.Dropout(p=0.4)
        # self.fc1_drop1 = nn.Dropout(p=0.1)
        # self.fc1_drop2 = nn.Dropout(p=0.2)
        # self.fc1_drop3 = nn.Dropout(p=0.3)
        # self.fc1_drop4 = nn.Dropout(p=0.4)
        # self.fc1_drop5 = nn.Dropout(p=0.5)
        # self.fc1_drop6 = nn.Dropout(p=0.6)
        # self.fc2 = nn.Linear(1024, 136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        # Another attempt, this time smaller images. It makes checkpoint
        # files much smaller as well.

        self.conv1 = nn.Conv2d(1, 16, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 512, 3)
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.fc1_drop = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(1024, 136)  

        # self.conv1 = nn.Conv2d(1, 8, 7)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(8, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 3)
        # self.conv4 = nn.Conv2d(32, 64, 3)
        # self.conv5 = nn.Conv2d(64, 512, 3)
        # self.fc1 = nn.Linear(512*4*4, 1024)
        # self.fc1_drop = nn.Dropout(p=0.6)
        # self.fc2 = nn.Linear(1024, 136) 

        # self.conv1 = nn.Conv2d(1, 8, 7)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(8, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 3)
        # self.conv4 = nn.Conv2d(32, 64, 3)
        # self.conv5 = nn.Conv2d(64, 128, 3)
        # self.fc1 = nn.Linear(128*4*4, 512)
        # self.fc1_drop1 = nn.Dropout(p=0.1)
        # self.fc1_drop2 = nn.Dropout(p=0.2)
        # self.fc1_drop3 = nn.Dropout(p=0.3)
        # self.fc1_drop4 = nn.Dropout(p=0.4)
        # self.fc1_drop5 = nn.Dropout(p=0.5)
        # self.fc1_drop = nn.Dropout(p=0.6)
        # self.fc2 = nn.Linear(512, 136) 
        # self.dense1_bn = nn.BatchNorm1d(512)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = self.fc2(x)

        # # Basing architecture off NaimishNet paper
        # # https://arxiv.org/pdf/1710.00977.pdf
        # #
        # # The paper used Exponential Linear Units (ELU) as activation function
        # # Dropoff after every maxpooling. It increased from 0.1 to 0.6
        # # x.view() is flatten in pytorch
        # x = self.pool(F.elu(self.conv1(x)))
        # x = self.fc1_drop1(x)
        # x = self.pool(F.elu(self.conv2(x)))
        # x = self.fc1_drop2(x)
        # x = self.pool(F.elu(self.conv3(x)))
        # x = self.fc1_drop3(x)
        # x = self.pool(F.elu(self.conv4(x)))
        # x = self.fc1_drop4(x)
        # x = x.view(x.size(0), -1)
        # x = F.elu(self.fc1(x))
        # x = self.fc1_drop5(x)
        # x = self.fc2(x)
        
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.fc1_drop(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.fc1_drop(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.fc1_drop(x)
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.fc1_drop(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc1_drop(x)
#         x = self.fc2(x)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # x = self.pool(F.elu(self.conv1(x)))
        # x = self.fc1_drop1(x)
        # x = self.pool(F.elu(self.conv2(x)))
        # x = self.fc1_drop2(x)
        # x = self.pool(F.elu(self.conv3(x)))
        # x = self.fc1_drop3(x)
        # x = self.pool(F.elu(self.conv4(x)))
        # x = self.fc1_drop4(x)
        # x = self.pool(F.elu(self.conv5(x)))
        # x = self.fc1_drop5(x)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = F.relu(self.dense1_bn(x))
        # x = self.fc2(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x
