from .parts_top_view_AE import Autoencoder
from .module_utils import Flatten
from .module_utils import DeFlatten

import torch
from torchvision.transforms import functional as F

import os
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncodingsDataset():
    def __init__(self, artifact_dir, model_file, view, kind, transform):
        self.artifact_dir = artifact_dir
        self.view = view
        self.transform = transform
        
        self.camera_images_dir = os.path.join(artifact_dir, 'data', 'camera_data', kind)
        self.parts_images_dir = os.path.join(artifact_dir, 'data', 'parts_data_2', kind)

        self.model_dir = os.path.join(artifact_dir, 'models')
        self.model_path = os.path.join(self.model_dir, model_file)

        self.camera_image_names = self._get_camera_images_by_view(self.camera_images_dir, view)
        self.parts_image_names = self._get_camera_images_by_view(self.parts_images_dir, view)
        
        self.encoder = torch.load(self.model_path).encoder.to(torch.device('cpu'))
        self.encoder = self.encoder.eval()
    
    def __getitem__(self, idx):
        assert self.camera_image_names[idx] == self.parts_image_names[idx]

        camera_image_path = os.path.join(self.camera_images_dir, self.camera_image_names[idx] + '.jpeg')
        parts_image_path = os.path.join(self.parts_images_dir, self.camera_image_names[idx] + '.png')
        model_path = os.path.join(self.model_dir, '')

        camera_image = Image.open(camera_image_path).convert('RGB')
        camera_image = F.to_tensor(camera_image) 

        parts_image = Image.open(parts_image_path).convert('RGB')
        parts_image = F.to_tensor(parts_image)

        parts_image = self.transform(parts_image)
        parts_image_1 = parts_image.view(1, *parts_image.shape)
        parts_image_1 = parts_image_1.to(torch.device('cpu'))
        with torch.no_grad():
            part_encoding = self.encoder(parts_image_1)
        
        part_encoding.view(part_encoding.shape[1])
        
        return self.transform(camera_image), part_encoding

    def __len__(self):
        return len(self.camera_image_names)

    def _get_camera_images_by_view(self, path, view):
        images = list(sorted(os.listdir(path)))
        l = map(lambda name: name.split('.')[0], images)
        l = list(sorted(list(filter(lambda name: name.endswith(view), l))))
        return l
        
