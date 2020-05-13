import os
import random
import time
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

from data_helper import UnlabeledDataset
from helper import collate_fn, draw_box
from ResNet import ResNet


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def generate_rotation_data(images, post_transforms):

    images_tensor = []
    expected_outputs =[]
    for image in images:
        # perform rotation
        image = image.resize((64, 64))
        
        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_image = image.rotate(rot_angle)

        sample = post_transforms(rot_image)
        
        images_tensor.append(sample)
        
        expected_outputs.append(rot_class)
        
    images = torch.stack(images_tensor)
    expected_outputs = torch.LongTensor(expected_outputs)

    return images, expected_outputs


def train_model(model, dataloaders, data_generator, train_post_transforms, criterion, optimizer, device, checkpoint_path, f, verbIter, num_epochs=25, scheduler = None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        f.write('Epoch {}/{} \n'.format(epoch, num_epochs - 1))
        f.write('-' * 10)
        f.write('\n')
        f.flush()
        
        np.random.seed()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            end = time.time()

            # Iterate over data.
            for batch_num, (images, indexes) in enumerate(dataloaders[phase]):
                
                inputs, labels = data_generator(images, train_post_transforms)

                data_time = time.time() - end
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                batchSize = inputs.size(0)
                n_samples += batchSize

                # forward
                # track history if only in train
                forward_start_time  = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                forward_time = time.time() - forward_start_time

                # statistics
                running_loss += loss.item() * inputs.size(0)
                pred_top_1 = torch.topk(outputs, k=1, dim=1)[1]
                running_corrects += pred_top_1.eq(labels.view_as(pred_top_1)).int().sum().item()
                
                if batch_num % verbIter == 0:
                    # Metrics
                    top_1_acc = running_corrects/n_samples
                    epoch_loss = running_loss / n_samples

                    f.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))
                    f.write('Full Batch time: {} , Data load time: {} , Forward time: {}\n'.format(time.time() - end, data_time, forward_time))
                    f.flush()

                end = time.time()
                
            # Metrics
            top_1_acc = running_corrects/n_samples
            epoch_loss = running_loss / n_samples

            f.write('{} Loss: {:.4f} Top 1 Acc: {:.4f} \n'.format(phase, epoch_loss, top_1_acc))
            f.flush()

            # deep copy the model
            if phase == 'val' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), '%s/net_epoch_%d.pth' % (checkpoint_path, epoch))

    time_elapsed = time.time() - since
    f.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('Best val Acc: {:4f} \n'.format(best_acc))
    f.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = '/home/prs392/codes/top_view_predictor/artifacts/data'

    image_folder = data_dir
    annotation_csv = f'{data_dir}/annotation.csv'
        

    batch_size = 32

    train_pre_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip()
    ])

    val_pre_transforms = torchvision.transforms.Compose([
    ])

    train_post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_post_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    train_set = UnlabeledDataset(
        image_folder = image_folder,
        scene_index = np.arange(0, 101),
        first_dim = 'image',
        transform = train_pre_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    val_set = UnlabeledDataset(
        image_folder = image_folder,
        scene_index = np.arange(101, 106),
        first_dim = 'image',
        transform = val_pre_transforms
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

#     count = 0
#     for i, data in enumerate(train_loader):
#         images, targets = data
#         samples, expected_outputs = generate_rotation_data(images, train_post_transforms)

#         count += 1
#         break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    fc_layer = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 4),
                                    nn.LogSoftmax(dim=1))

    model = ResNet(pretrained = False, ssl_pretrained = False, ssl_pretrained_dict_path = '', fc_layer=fc_layer).model
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Sequential(nn.Linear(512, 512),
#                                     nn.ReLU(),
#                                     nn.Dropout(0.2),
#                                     nn.Linear(512, 4),
#                                     nn.LogSoftmax(dim=1))
    model = model.to(device)

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    criterion = nn.NLLLoss()
    optimizer_conv = optim.Adam(model.parameters(), lr=0.003)

    f = open("{}/training_logs.txt".format('.'), "w+")

    model_ft = train_model(model, dataloaders, generate_rotation_data, train_post_transforms, criterion, optimizer_conv, device, '/scratch/prs392/models/rotation_ssl', f, 100, num_epochs=50)


if __name__ == "__main__":
    main()