

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
This is a parent class of a model that we would like to init during our pipelines.
"""


class BaseModel(nn.Module):
    """
    init config file here.
    """
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)


    """
    Inherited methods for subclasses
    """
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass



    @abstractmethod
    def evaluate(self):
        pass
