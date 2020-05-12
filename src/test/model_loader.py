"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import utils
from torch.autograd import Variable
from torchvision import models
import PIL.ImageOps
from PIL import Image
import numpy as np
from math import *


def get_results_img(x, nrow=8, padding=5, cnn = True):
    if cnn:
        return utils.make_grid(
            x.view(-1, *(3, 256, 306)),
            nrow=nrow, padding=padding)
    else:
        return utils.make_grid(
            x.view(-1, *(3, 128, 128)),
            nrow=nrow, padding=padding)


normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class DeFlatten(nn.Module):
    def __init__(self, *args):
        super(DeFlatten, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.8),
            nn.Conv2d(96, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(8, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.8),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(96, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            Flatten(),
            nn.Linear(in_features=8192, out_features=1024, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=64, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=64, out_features=1024, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=8192, bias=True),
            DeFlatten(-1, 8, 32, 32),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(8, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1), dilation = (1,1), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(8, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding = (1, 1), dilation = (1,1), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(96, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        noise = Variable(torch.randn(x.size()) * 0.3)
        noise = noise.to(self.device)
        x = self.decoder(x + noise)
        return x


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


inv_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    transformation_compose = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) ])
    return transformation_compose


a = 64
from math import *

def between_angles(y, x, theta1, theta2):
    return degrees(atan2(y, x)) >= theta1 and degrees(atan2(y, x)) <= theta2

def in_front(x, y):
    return x >= 0 and x <= a and degrees(atan2(y, x)) <= 35 and degrees(atan2(y, x)) >= -35

def in_front_left(x, y):
    if x == 0:
        return y >= 0
    
    return x <= a and y <= a and between_angles(y, x, 25, 95)

def in_front_right(x, y):
    if x == 0:
        return y <= 0
    
    return y <= 0 and degrees(atan2(y, x)) <= -25 and degrees(atan2(y, x)) >= -95

def in_back(x, y):
    return x <= 0 and (between_angles(y, x, 145, 180) or between_angles(y, x, -180, -145))

def in_back_left(x, y):
    if x == 0:
        return y >= 0
    
    return y >= 0 and between_angles(y, x, 85, 155)

def in_back_right(x, y):
    if x == 0:
        return y <= 0
    
    return y <= 0 and between_angles(y, x, -155, -85)

def find_regions(x, y):
    regions = []

    if in_front_left(x, y):
        regions.append(1)
    if in_front(x, y):
        regions.append(2)
    if in_front_right(x, y):
        regions.append(3)
    if in_back_left(x, y):
        regions.append(4)
    if in_back(x, y):
        regions.append(5)
    if in_back_right(x, y):
        regions.append(6)
    
    return regions

def region_picker(region1, region2):
    resolve_map = {
        (1, 2): 1,
        (2, 3): 2,
        (4, 5): 5,
        (5, 6): 6,
        (1 ,4): 4,
        (3, 6): 3
    }

    return resolve_map[(region1, region2)]

def resolve_color(x, y, images):
    regions = find_regions(x - a, -y + a)
    # print(regions)
    if len(regions) == 1:
        region = regions[0]
        return images[region - 1].getpixel((x, y))
    
    return images[region_picker(regions[0], regions[1]) - 1].getpixel((x, y))

def stitch_images(images):
    front_left = torchvision.transforms.ToPILImage(mode = None)(images[0]).convert('RGB')
    front = torchvision.transforms.ToPILImage(mode = None)(images[1]).convert('RGB')
    front_right = torchvision.transforms.ToPILImage(mode = None)(images[2]).convert('RGB')
    back_left = torchvision.transforms.ToPILImage(mode = None)(images[3]).convert('RGB')
    back = torchvision.transforms.ToPILImage(mode = None)(images[4]).convert('RGB')
    back_right = torchvision.transforms.ToPILImage(mode = None)(images[5]).convert('RGB')

    images_pil = [
        front_left,
        front,
        front_right,
        back_left,
        back,
        back_right
    ]

    ego = Image.new('RGB', front.size, (0,0,0))

    a = 64

    width, height = front.size
    for x in range(width):
        for y in range(height):
            c = resolve_color(x, y, images_pil)
            ego.putpixel((x, y), c)
    
    ego = ego.resize((800, 800))
    
    #### FOLLOWING LINE COMMENTED AS EXPLAINED IN THE EMAIL
    # ego = PIL.ImageOps.invert(ego)
    
    ego = ego.convert('L')
    ego_bin = Image.fromarray(binarize_array(np.array(ego), 200))
    
    # ego_bin.save('ego_image.png')
    
    ego = torchvision.transforms.functional.to_tensor(ego_bin)
    
    ego = ego.view(800, 800)
    
    return ego

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    round_number = 1
    team_member = ['Param Shah', 'Nikhil Supekar', 'Aajan Quail']
    contact_email = 'prs392@nyu.edu'

    def __init__(self, model_file='birds_eye_view_predictor.pt'):

        checkpoint = torch.load(model_file)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.front_cnn = models.resnet18()
        self.front_cnn.fc = nn.Linear(512, 64)
        self.front_cnn.load_state_dict(checkpoint['cnn_front'])
        self.front_cnn = self.front_cnn.to(self.device)
        self.front_cnn = self.front_cnn.eval()

        self.front_left_cnn = models.resnet18()
        self.front_left_cnn.fc = nn.Linear(512, 64)
        self.front_left_cnn.load_state_dict(checkpoint['cnn_front_left'])
        self.front_left_cnn = self.front_left_cnn.to(self.device)
        self.front_left_cnn = self.front_left_cnn.eval()

        self.front_right_cnn = models.resnet18()
        self.front_right_cnn.fc = nn.Linear(512, 64)
        self.front_right_cnn.load_state_dict(checkpoint['cnn_front_right'])
        self.front_right_cnn = self.front_right_cnn.to(self.device)
        self.front_right_cnn = self.front_right_cnn.eval()

        self.back_cnn = models.resnet18()
        self.back_cnn.fc = nn.Linear(512, 64)
        self.back_cnn.load_state_dict(checkpoint['cnn_back'])
        self.back_cnn = self.back_cnn.to(self.device)
        self.back_cnn = self.back_cnn.eval()

        self.back_left_cnn = models.resnet18()
        self.back_left_cnn.fc = nn.Linear(512, 64)
        self.back_left_cnn.load_state_dict(checkpoint['cnn_back_left'])
        self.back_left_cnn = self.back_left_cnn.to(self.device)
        self.back_left_cnn = self.back_left_cnn.eval()

        self.back_right_cnn = models.resnet18()
        self.back_right_cnn.fc = nn.Linear(512, 64)
        self.back_right_cnn.load_state_dict(checkpoint['cnn_back_right'])
        self.back_right_cnn = self.back_right_cnn.to(self.device)
        self.back_right_cnn = self.back_right_cnn.eval()

        self.ae  = Autoencoder()
        self.ae.load_state_dict(checkpoint['ae'])
        self.ae = self.ae.to(self.device)
        self.ae = self.ae.eval()

        self.encoder = self.ae.encoder
        self.decoder = self.ae.decoder

        self.decoder = self.decoder.to(self.device)
        self.encoder = self.encoder.to(self.device)


    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 

        camera_image_front_left = samples[:, 0]
        camera_image_front_left = camera_image_front_left.to(self.device)

        camera_image_front = samples[:, 1]
        camera_image_front = camera_image_front.to(self.device)

        camera_image_front_right = samples[:, 2]
        camera_image_front_right = camera_image_front_right.to(self.device)

        camera_image_back_left = samples[:, 3]
        camera_image_back_left = camera_image_back_left.to(self.device)

        camera_image_back = samples[:, 4]
        camera_image_back = camera_image_back.to(self.device)
        
        camera_image_back_right = samples[:, 5]
        camera_image_back_right = camera_image_back_right.to(self.device)

        with torch.no_grad():
            
            # FRONT
            # print(camera_image_front.shape)
            front_embedding = self.front_cnn(camera_image_front)
            front_embedding = front_embedding.to(self.device)
            front_decoder_output = self.decoder(front_embedding)

            front_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in front_decoder_output]), cnn=False
                ),
                0, 
                1
            )

            # FRONT - LEFT

            front_left_embedding = self.front_left_cnn(camera_image_front_left)
            front_left_embedding = front_left_embedding.to(self.device)
            front_left_decoder_output = self.decoder(front_left_embedding)

            front_left_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in front_left_decoder_output]), cnn=False
                ),
                0, 
                1
            )

            # FRONT - RIGHT

            front_right_embedding = self.front_right_cnn(camera_image_front_right)
            front_right_embedding = front_right_embedding.to(self.device)
            front_right_decoder_output = self.decoder(front_right_embedding)

            front_right_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in front_right_decoder_output]), cnn=False
                ),
                0, 
                1
            )

            # back

            back_embedding = self.back_cnn(camera_image_back)
            back_embedding = back_embedding.to(self.device)
            back_decoder_output = self.decoder(back_embedding)

            back_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in back_decoder_output]), cnn=False
                ),
                0, 
                1
            )

            # back - LEFT

            back_left_embedding = self.back_left_cnn(camera_image_back_left)
            back_left_embedding = back_left_embedding.to(self.device)
            back_left_decoder_output = self.decoder(back_left_embedding)

            back_left_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in back_left_decoder_output]), cnn=False
                ),
                0, 
                1
            )

            # back - RIGHT

            back_right_embedding = self.back_right_cnn(camera_image_back_right)
            back_right_embedding = back_right_embedding.to(self.device)
            back_right_decoder_output = self.decoder(back_right_embedding)

            back_right_normalized = torch.clamp(
                get_results_img(
                    torch.stack([inv_normalize(img1.detach()) for img1 in back_right_decoder_output]), cnn=False
                ),
                0, 
                1
            )
            
            output_tensor = torch.randn(samples.shape[0], 800, 800)
            
            for image_index in range(samples.shape[0]):
                
                # print('debug - ' + str(front_normalized.shape))
                front = front_normalized[image_index]
                front_left = front_left_normalized[image_index]
                front_right = front_right_normalized[image_index]
                back_left = back_left_normalized[image_index]
                back_right = back_right_normalized[image_index]
                back = back_normalized[image_index]
                
                ego = stitch_images([front_left, front, front_right, back_left, back, back_right])
                
                output_tensor[image_index] = ego
        
        return output_tensor