from resnet import ResNet
from autoencoder import Autoencoder
from module_utils import Flatten
from module_utils import DeFlatten
from helper import collate_fn
from data_helper import LabeledDataset
from resnet import ResNet
from utils_image import *

import torch
import torchvision.datasets as datasets
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from optparse import OptionParser
import os
import pickle
import numpy as np
from math import tan
from math import radians

from PIL import Image
from PIL import ImageDraw

parser = OptionParser()
parser.add_option('--batch-size', dest = 'batch_size', default = 64, type = 'int', help = 'batch size to process data')
parser.add_option('--num-workers', dest = 'num_workers', default = 2, type = 'int', help = 'GPU workers')
parser.add_option('--learning-rate', dest = 'learning_rate', default = 1e-3, type = 'float', help = 'learning rate')
parser.add_option('--num-epochs', dest = 'num_epochs', default = 50, type = 'int', help = 'number of epochs')
parser.add_option('--train-views', dest = 'train_views', default = '', type = 'string', help = 'views to be trained (comma separated)')
parser.add_option('--ae-model', dest = 'ae_model_path', default = '', type = 'string', help = 'model file for the trained autoencoder')
parser.add_option('--data-directory', dest = 'data_directory', default = '.', type = 'string', help = 'data path')
parser.add_option('--model-directory', dest = 'model_directory', default = '.', type = 'string', help = 'model save path')

(options, args) = parser.parse_args()

num_epochs = options.num_epochs
batch_size = options.batch_size
workers = options.num_workers
ae_model_path = options.ae_model_path
learning_rate = options.learning_rate
train_views = options.train_views.replace(' ', '').split(',')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet(
    pretrained = False,
    ssl_pretrained = False,
    ssl_pretrained_dict_path = None,
    fc_layer = nn.Linear(512, 64)
).model
model = model.to(device)
criterion = nn.MSELoss()


model_directory = options.model_directory
image_folder = options.data_directory
annotation_csv = os.path.join(image_folder, 'annotation.csv')
train_scene_index = np.arange(106, 131)
val_scene_index = np.arange(131, 134)

transform = torchvision.transforms.ToTensor()
train_dataset = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=train_scene_index,
    transform=transform,
    extra_info=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn
)

val_dataset = LabeledDataset(
    image_folder=image_folder,
    annotation_file=annotation_csv,
    scene_index=val_scene_index,
    transform=transform,
    extra_info=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn
)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

model_ae_encoder = Autoencoder()
model_ae_encoder.load_state_dict(torch.load(ae_model_path))
model_ae_encoder = model_ae_encoder.encoder.to(device)
model_ae_encoder = model_ae_encoder.eval()


for view in train_views:

    dataset_len = len(train_loader.dataset)
    val_dataset_len = len(val_loader.dataset)
    validation_losses = []
    running_avg_training_losses = []
    views = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        total = 0
        running_total_training_loss = 0

        print(f'-- running epoch {epoch + 1} --')

        for sample, target, road_image, extra in train_loader:
            sample = torch.stack(sample)
            camera_image = sample[:, views.index(view), :, :, :]
            masked_road_image_tensor = mask_road_image_by_view(road_image, view, (800, 800))
            masked_road_image_tensor = masked_road_image_tensor[:, 0, :, :, :]
            masked_road_image_tensor = masked_road_image_tensor.to(device)

            with torch.no_grad():
                part_encoding = model_ae_encoder(masked_road_image_tensor)

            img = camera_image
            expected_output = part_encoding
            
            img = img.to(device)
            expected_output = expected_output.to(device)
            
            output = model(img) 
            loss = criterion(output, expected_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += 1    

            running_total_training_loss += float(loss)    

        running_avg_training_losses.append(running_total_training_loss/total)

        with torch.no_grad():
            total_vloss = 0
            for sample, target, road_image, extra in val_loader:
                sample = torch.stack(sample)
                camera_image = sample[:, views.index(view), :, :, :]
                masked_road_image_tensor = mask_road_image_by_view(road_image, view, (800, 800))
                masked_road_image_tensor = masked_road_image_tensor[:, 0, :, :, :]
                masked_road_image_tensor = masked_road_image_tensor.to(device)

                with torch.no_grad():
                    part_encoding = model_ae_encoder(masked_road_image_tensor)

                img = camera_image
                expected_output = part_encoding
                
                img = img.to(device)
                expected_output = expected_output.to(device)

                voutput = model(img)
                vloss = criterion(voutput, expected_output)
                total_vloss += vloss
            validation_losses.append(total_vloss)


        print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, running avg training loss:{running_avg_training_losses[-1]:.4f}')

        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(view + '_cnn_'+ str(epoch + 1) +'.pt'))

