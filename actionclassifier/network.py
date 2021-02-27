"""
Module contains class NeuralNetwork which defines a neural network to predict image labels for actions in sports

Author: Joe Nunn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Neural network to predict probabilities of human actions in sports being shown in images.
    """
    def __init__(self, num_classes):
        """
        Initialises layers of the neural network

        :param num_classes: number of classes used
        """
        super().__init__()
        # maintains size of 128x128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # maintains size of 64x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Half size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Original image of size 128x128 is halved twice to 32 * 32. With 64 channels flattened is 32 * 32 * 64
        self.fc1 = nn.Linear(32 * 32 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, batch):
        """
        Calculates predictions of labels for images in the batch

        :param batch: batch of images as tensors
        :return: tensor of predictions of labels of the batch
        """
        # Convolutional layers
        batch = self.conv1(batch)
        batch = F.relu(batch)
        batch = self.pool(batch)
        batch = self.conv2(batch)
        batch = F.relu(batch)
        batch = self.pool(batch)
        # Flatten
        batch = batch.reshape(batch.shape[0], -1)
        # Fully connected layers
        batch = self.fc1(batch)
        batch = self.dropout(batch)
        batch = self.fc2(batch)
        batch = torch.sigmoid(batch)
        return batch
