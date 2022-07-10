import torch
#from torch.utils.data import DataLoader


class DataLoader:
    @staticmethod
    def load_data(data_config):
        """
        Loads data from the path
        """
        return torch.utils.data.DataLoader(dataset=data_config.path) 
