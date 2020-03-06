#!/usr/bin/env python3
"""Define the neural network model by extending torch.nn.Module class
"""

import numpy as np
import torch.nn as tnn
import torch.nn.functional as F

from model.triplet_loss import batch_hard_triplet_loss


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network
    """
    def __init__(self, params):
        """Initialize the different layers in the neural network
        Args:
            params: (Params) contains hyperparameters
        """
        super(Net, self).__init__()

        self.conv1 = tnn.Conv2d(1, params.num_channels, 3, stride=1, padding=1)
        self.conv2 = tnn.Conv2d(params.num_channels,
                                params.num_channels*2, 3, stride=1, padding=1)

        self.fc1 = tnn.Linear(params.num_channels*2*7*7, params.num_channels)
        self.dropout_rate = params.dropout_rate

    def forward(self, x):
        """Defines the forward propagation through the network
        Args:
            x: (torch.Tensor) contains a batch of images.
        Returns:
            out: (torch.Tensor) Embeddings of the input (batch_size, num_channels)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        out = x.view(x.size(0), -1)

        return out


def loss_fn(outputs, ground_truth, params):
    """Compute the loss given outputs and ground_truth.
    Args:
        outputs: (torch.Tensor) output of network forward pass
        ground_truth: (torch.Tensor) batch of ground truth
        params: (Params) contains hyperparameters
    Returns:
        loss: (torch.Tensor) loss for all the inputs in the batch
    """
    loss = batch_hard_triplet_loss(ground_truth, outputs, margin=params.margin)
    return loss


def accuracy(outputs, labels):
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the network
        labels: (np.ndarray) ground truth labels
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


# Maintain all metrics required during training and evaluation.
def get_metrics():
    """Returns a dictionary of all the metrics to be used
    """
    metrics = {}
    return metrics
