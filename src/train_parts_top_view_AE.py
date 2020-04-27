from modules.parts_top_view_AE import Autoencoder
from modules.module_utils import NormalizeInverse

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

import numpy as np

np.random.seed(0)
torch.manual_seed(0)

data = "parts_data"
batch_size = 64
workers = 4
distributed = False

train_dir = os.path.join(data, 'train')
val_dir = os.path.join(data, 'val')

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]))

if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir, transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
criterion = nn.MSELoss()

learning_rate = 1e-3

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

num_epochs = 50
dataset_len = len(train_loader.dataset)
val_dataset_len = len(val_loader.dataset)
validation_losses = []
running_avg_training_losses = []

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    total = 0
    running_total_training_loss = 0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        # ===================forward=====================
        output = model(img) 
        loss = criterion(output, img.data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total += 1    

        running_total_training_loss += loss.item()

        if len(validation_losses) == 0:
            print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, running avg training loss:{running_total_training_loss / total:.4f}')
        else:
            print(f'epoch [{epoch + 1}/{num_epochs}], data trained:{100 * total / dataset_len :.3f}%, running avg training loss:{running_total_training_loss / total:.4f}, validation loss (prev epoch):{validation_losses[-1]}')
    
    running_avg_training_losses.append(running_total_training_loss/total)

    with torch.no_grad():
        total_vloss = 0
        for val_data in val_loader:
            vimg, _ = val_data
            vimg = vimg.to(device)
            voutput = model(vimg)
            vloss = criterion(voutput, vimg.data)
            total_vloss += vloss.item()
        validation_losses.append(total_vloss)

    if (epoch + 1) % 10 == 0:
        if torch.cuda.is_available():
            torch.save(model, 'Models/ae_latent_noise_gpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
            model.to(torch.device('cpu'))
            torch.save(model, 'Models/ae_latent_noise_cpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')
            model.to(device)   
        else:
            torch.save(model, 'Models/ae_latent_noise_cpu_model_b64_w2_e'+ str(epoch + 1) +'.pt')