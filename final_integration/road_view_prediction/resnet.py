import torch
import torch.nn as nn
import torchvision.models as models


class ResNet():

    def __init__(self, pretrained = False, ssl_pretrained = False, ssl_pretrained_dict_path = '', fc_layer = None):
        if pretrained == True and ssl_pretrained == True:
            assert False, "pretrained and ssl_pretrained cannot both be True"
        self.model = models.resnet18(pretrained=pretrained)

        if fc_layer is not None:
            self.model.fc = fc_layer

        if ssl_pretrained == True:
            assert ssl_pretrained_dict_path != ''

            pretrained_dict = torch.load(ssl_pretrained_dict_path)
            model_dict = self.model.state_dict()

            for k, v in pretrained_dict.items():
                if k in model_dict:
                    model_dict.update({k: v})

            self.model.load_state_dict(model_dict)