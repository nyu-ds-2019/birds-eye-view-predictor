# Import some libraries
from modules.parts_top_view_AE import Autoencoder
from modules.encodings_dataset import EncodingsDataset
from modules.module_utils import Flatten
from modules.module_utils import DeFlatten
from torchvision import models

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

import os
import pickle
import numpy as np

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = nn.Linear(512, 64)
model = model.to(device)
criterion = nn.MSELoss()


batch_size = 32
workers = 2

train_dataset = EncodingsDataset(
    '../artifacts',
    'ae_latent_noise_gpu_model_b64_w2_e20.pt',
    'front',
    'train',
    transforms.Compose(
        [
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            )
        ]
    )
)

val_dataset = EncodingsDataset(
    '../artifacts',
    'ae_latent_noise_gpu_model_b64_w2_e20.pt',
    'front',
    'val',
    transforms.Compose(
        [
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ]
    )
)


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

learning_rate = 1e-1

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


num_epochs = 300
dataset_len = len(train_loader.dataset)
val_dataset_len = len(val_loader.dataset)
validation_losses = []
running_avg_training_losses = []

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    total = 0
    running_total_training_loss = 0

    print(f'-- running epoch {epoch + 1} --')

    for data in train_loader:
        img, expected_output = data
        img = img.to(device)
        expected_output = expected_output.to(device)
        expected_output = expected_output.view(expected_output.shape[0], expected_output.shape[2])
        # ===================forward=====================
        output = model(img) 
        loss = criterion(output, expected_output)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += 1    

        running_total_training_loss += float(loss)    
    running_avg_training_losses.append(running_total_training_loss/total)
    print('in val')
    with torch.no_grad():
        total_vloss = 0
        for val_data in val_loader:
            vimg, v_expected_output = val_data
            v_expected_output = v_expected_output.view(v_expected_output.shape[0], v_expected_output.shape[2])
            vimg = vimg.to(device)
            v_expected_output = v_expected_output.to(device)
            voutput = model(vimg)
            vloss = criterion(voutput, v_expected_output)
            total_vloss += float(vloss)
        validation_losses.append(total_vloss / val_dataset_len)


    print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, running avg training loss:{running_avg_training_losses[-1]:.4f}')
    print(validation_losses)
    
    if (epoch + 1) % 10 == 0:
        if torch.cuda.is_available():
            torch.save(model, '../artifacts/models/cnn_latent_noise_gpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
            model.to(torch.device('cpu'))
            torch.save(model, '../artifacts/models/cnn_latent_noise_cpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
            model.to(device)   
        else:
            torch.save(model, '../artifacts/models/cnn_latent_noise_cpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')

