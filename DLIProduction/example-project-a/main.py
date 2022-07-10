from configs.config import CFG
from model.unet import UNet


def dorun(): 
    """builds the model, loads data, trains and evaluates"""
    model = UNet(CFG) # <- pass the configuration schema here.
    model.load_data()
    model.build()
    model.train()
    model.evaluate()

if __name__ == "__main__":
    dorun()
