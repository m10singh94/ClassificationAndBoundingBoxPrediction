"""
Contains DatasetGenrator and creates a dataset
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import os
import pathlib
from PIL import Image

class CustomDatasetGenerator(Dataset):
  def __init__(self, tensors, transform=None):
    self.tensors = tensors
    self.transform = transform

  def __len__(self):
	  return len(self.tensors[0])

  def __getitem__(self, index):
    image = self.tensors[0][index]
    label = self.tensors[1][index]
    bbox = self.tensors[2][index]

    image = image.permute(2,0,1)
    if self.transform:
      image = self.transform(image)

    return (image, label, bbox)
