"""
This file contains the function to create and return an instance of our model
"""

import torch
import torchvision
from torch import nn
import objectdetector

def create_model(num_classes:int=2,
                seed:int=42):
    """Creates a Resnet feature extractor model.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 2.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): Resnet feature extractor model.
    """
    # downloading weights and creating teh base model
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    resnet = torchvision.models.resnet50(weights=weights)

    for param in resnet.parameters():
        param.requires_grad = False

    objectDetector = ObjectDetector(baseModel=resnet,
                                # in_features=resnet.fc.in_features,
                                numClasses=num_classes)
    return objectDetector
