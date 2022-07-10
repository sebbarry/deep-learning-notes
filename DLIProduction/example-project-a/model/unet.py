"""
UNet Model.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


from .base_model import BaseModel
from dataloader.dataloder import DataLoader

class UNet(BaseModel):

    def __init__(self, config): 
        # init superclass config.
        super().__init__(config)

        # TODO make base model here.
        #self.base_model = nn.Sequential...
        

        self.model = None
        self.output_channels = self.config.model.output
        self.dataset = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epochs = self.config.train.epochs
        self.val_subsplits = self.config.train.val_subscripts
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.up_stack = self.config.model.up_stack


        self.image_size = self.config.data.image_size
        self.train_dataset = []
        self.test_dataset = []


    def load_data(self): 
        self.dataset, self.info = DataLoader().load_data(
                self.config.data
                )
        # preprocess the data
        self._preprocess_data()

    def _preprocess_data(self):
        # prepare the data here
        pass

    def _set_training_parameters(self):
        # set the training parameters for the model.
        pass

    def _normalize(self, input_image, input_mask):
        # normalize the data somehow ie. x/255
        pass

    def _load_image_train(self, datapoint):
        # load the training data
        pass

    def _load_image_test(self, datapoint): 
        # load the test data
        pass

    def build(self):
        # build the model here.
        self.encoder = Encoder(self.up_stack)
        self.decoder = Decoder(self.up_stack[::-1]) # reverse the vlaues.
        self.head = nn.Conv2d(self.up_stack[-1], 1,1)
        self.retain_dim = retain_dim


    def train(self):
        # Train the model.
        pass

    def evaluate(self):
        # evalute the model.
        pass






"""

"The contractive path consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels."

"""

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)


    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))



class Encoder(nn.Module):

    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            Block(chs[i], chs[i+1]) for i in range(len(chs)-1)
            ])
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks: 
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


    
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


