

"""
Config class
"""

import json

class Config:
    """
    Config class - this  contains the
    1. data paths
    2. hyperparameters
    3. structural layout schema
    """
    def __init__(self, data, train, model):
        self.data = data
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Create the json from teh config"""
        params = json.loads(
                json.dumps(cfg), object_hook=HelperObject
                )
        return cls(params.data, params.train, params.model)


class HelperObject(object):
    """
    Helper class to convert json into python object
    """
    def __init__(self, dict_):
        self.__dict__.update(dict_)

