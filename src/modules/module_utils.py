import torch
from torch import nn
import torchvision

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class DeFlatten(nn.Module):
    def __init__(self, *args):
        super(DeFlatten, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


#### TODO: use global variable
inv_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])