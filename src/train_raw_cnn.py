# Import some libraries

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

from modules.parts_top_view_AE import Autoencoder
from modules.raw_cnn import CNN 
from modules.encodings_dataset import EncodingsDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Autoencoder()
model.load_state_dict(torch.load('Models/s_gpu_model_b64_w2_e50.pt'))
encoder = model.encoder
decoder = model.decoder

decoder = decoder.to(device)
encoder = encoder.to(device)

model = CNN().to(device)
criterion = nn.MSELoss()


batch_size = 32
workers = 2

train_dataset = EncodingsDataset(
    'artifacts',
    'ae_latent_noise_gpu_model_b64_w2_e10.pt',
    'front',
    'train',
    transforms.Compose(
        [
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ]
    )
)

val_dataset = EncodingsDataset(
    'artifacts',
    'ae_latent_noise_gpu_model_b64_w2_e10.pt',
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


num_epochs = 50
dataset_len = len(train_loader.dataset)
val_dataset_len = len(val_loader.dataset)
validation_losses = []
training_losses = []
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    total = 0
    for data in train_loader:
        img, expected_output = data
        img = img.to(device)
        expected_output = expected_output.to(device)
        # ===================forward=====================
        output = model(img) 
#         output.view(output.shape[0], output.shape[2]).shape
        print(output.shape)
        print(expected_output.shape)
        loss = criterion(output, expected_output)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        clear_output(wait=True)
        total += len(data[0])     
        if len(validation_losses) == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, training loss:{loss.item():.4f}')
        else:
            print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, training loss:{loss.item():.4f}, validation loss (prev epoch):{validation_losses[-1]}')
    
#     with torch.no_grad():
#         total_vloss = 0
#         for val_data in val_loader:
#             vimg, v_expected_output = val_data
#             vimg = vimg.to(device)
#             voutput = model(vimg)
#             vloss = criterion(voutput, v_expected_output)
#             total_vloss += vloss
#         validation_losses.append(total_vloss)
        
#     with torch.no_grad():
#         total_tloss = 0
#         for train_data in train_loader:
#             timg, t_expected_output = train_data
#             timg = timg.to(device)
#             toutput = model(timg)
#             tloss = criterion(toutput, t_expected_output)
#             total_tloss += tloss
#         training_losses.append(total_tloss)

    if (epoch + 1) % 10 == 0:
        torch.save(model, 'artifacts/models/cnn_gpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
        model.to(torch.device('cpu'))
        torch.save(model, 'artifacts/models/cnn_cpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
        model.to(device)