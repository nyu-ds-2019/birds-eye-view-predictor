from autoencoder import Autoencoder
from helper import collate_fn
from data_helper import LabeledDataset
from utils_image import *

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from optparse import OptionParser

import torch
import torchvision
import torchvision.datasets as datasets

import os
from math import tan
from math import radians
import numpy as np
from PIL import ImageDraw


parser = OptionParser()
parser.add_option('--batch-size', dest = 'batch_size', default = 64, type = 'int', help = 'batch size to process data')
parser.add_option('--num-workers', dest = 'num_workers', default = 2, type = 'int', help = 'GPU workers')
parser.add_option('--learning-rate', dest = 'learning_rate', default = 1e-3, type = 'float', help = 'learning rate')
parser.add_option('--num-epochs', dest = 'num_epochs', default = 50, type = 'int', help = 'number of epochs')
parser.add_option('--random-seed', dest = 'random_seed', default = 0, type = 'int', help = 'random seed')
parser.add_option('--data-directory', dest = 'data_directory', default = '.', type = 'string', help = 'data path')
parser.add_option('--model-directory', dest = 'model_directory', default = '', type = 'string', help = 'model save path')

(options, args) = parser.parse_args()

RANDOM_SEED = options.random_seed

batch_size = options.batch_size
workers = options.num_workers
learning_rate = options.learning_rate
num_epochs = options.num_epochs

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model = model.to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

dataset_len = len(train_loader.dataset)
val_dataset_len = len(val_loader.dataset)
validation_losses = []
running_avg_training_losses = []
views = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']
best_vloss = 1e10
save_best_model = False

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    total = 0
    running_total_training_loss = 0

    print(f'-- running epoch {epoch + 1} --')

    for sample, target, road_image, extra in train_loader:
        for view in views:
            masked_road_image_tensor = mask_road_image_by_view(road_image, view, (800, 800))
            masked_road_image_tensor = masked_road_image_tensor.to(device)
            masked_road_image_tensor = masked_road_image_tensor[:, 0, :, :, :]
            output = model(masked_road_image_tensor)
            loss = criterion(output, masked_road_image_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += 1
            running_total_training_loss += float(loss)

    running_avg_training_losses.append(running_total_training_loss/total)

    with torch.no_grad():
        total_vloss = 0
        

        for sample, target, road_image, extra in val_loader:
            for view in views:
                masked_road_image_tensor = mask_road_image_by_view(road_image, view, (800, 800))
                masked_road_image_tensor = masked_road_image_tensor.to(device)
                masked_road_image_tensor = masked_road_image_tensor[:, 0, :, :, :]
                output = model(masked_road_image_tensor)
                vloss = criterion(output, masked_road_image_tensor)

                total_vloss += float(vloss)
        
        if total_vloss < best_vloss:
            print(f'Best performing validation model at {epoch}')
            best_vloss = total_vloss
            save_best_model = True
        
        validation_losses.append(total_vloss)
        
    print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, running avg training loss:{running_avg_training_losses[-1]:.4f}')
    print(validation_losses)
    
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(model_directory, 'ae'+ str(epoch + 1) +'.pt'))
        
    if save_best_model:
        torch.save(model.state_dict(), os.path.join(model_directory, 'ae_best_performing.pt'))
        print('Best performing AE model saved')
        save_best_model = False

