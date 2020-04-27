from .module_utils import Flatten
from .module_utils import DeFlatten

import torch
from torch import nn
from torch.autograd import Variable
import torchvision



class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.Dropout(p = 0.8),
            nn.Conv2d(96, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(8, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.Dropout(p = 0.8),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.Conv2d(96, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            Flatten(),
            nn.Linear(in_features=8192, out_features=1024, bias=True),
            ReLU(),
            nn.Linear(in_features=1024, out_features=64, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=1024, bias=True),
            ReLU(),
            nn.Linear(in_features=1024, out_features=8192, bias=True),
            DeFlatten(-1, 8, 32, 32),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(8, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1), dilation = (1,1), bias=False),
            ReLU(),
            nn.ConvTranspose2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.ConvTranspose2d(96, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(8, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1), dilation = (1,1), bias=False),
            ReLU(),
            nn.ConvTranspose2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            ReLU(),
            nn.ConvTranspose2d(96, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x, y):
        x = self.encoder(x)
        x = self.decoder(x + y)
        return x