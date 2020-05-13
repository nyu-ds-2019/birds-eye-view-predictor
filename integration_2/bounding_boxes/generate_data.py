from optparse import OptionParser
from math import tan, radians

from data_helper import UnlabeledDataset, LabeledDataset
from helper import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw

def to_PIL(image_tensor):
    image_tensor[image_tensor == 1] = 255
    return torchvision.transforms.ToPILImage(mode = None)(image_tensor.cpu()).convert('RGB')

def to_tensor(PIL_image):
    return torchvision.transforms.functional.to_tensor(PIL_image)

def get_masking_polygon_by_view(view, a):
    a = a/2
    polygon_map_centered_axes = {
        'front_left': [
            (0, 0),
            (a + (a / tan(radians(85))), 0),
            (a, a),
            (2 * a, a),
            (2 * a, 2 * a),
            (0, 2 * a)
        ],

        'front': [
            (0, 0),
            (2 * a, 0),
            (2 * a, - a * tan(radians(35)) + a),
            (a, a),
            (2 * a, a * tan(radians(35)) + a),
            (2 * a, 2 * a),
            (0, 2 * a)
        ],

        'front_right': [
            (0, 0),
            (2 * a, 0),
            (2 * a, a),
            (a, a),
            (a - (a / tan(radians(95))), 2 * a),
            (0, 2 * a)
        ],

        'back_left': [
            (a, 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (0, 2 * a),
            (0, -a * tan(radians(25)) + a),
            (a, a)
        ],

        'back': [
            (a, a),
            (0, -a * tan(radians(35)) + a),
            (0, 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (0, 2 * a),
            (0, a * tan(radians(35)) + a)
        ],

        'back_right': [
            (a, a),
            (0, a),
            (0, 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (a - a / tan(radians(85)), 2 * a)
        ]
    }
    
    return polygon_map_centered_axes[view]


def mask_bbox_image_by_view(road_image, view, size):
    '''
        masks the rest of the camera views in the top view

        road_image = PIL image
        view = front_left / front / front_right / back_left / back / back_right
        size = (h, h) of the road image 
    '''	

    polygon_coordinates = get_masking_polygon_by_view(view, size[0])
#     print(polygon_coordinates)
    road_image_masked = road_image.copy()

    drawer = ImageDraw.Draw(road_image_masked)
    drawer.polygon(polygon_coordinates, fill = (0, 0, 0))

    return road_image_masked


def draw_bboxes(image, bboxes, categories):
    
    image2 = image.copy()
    drawer = ImageDraw.Draw(image2)

    for i in range(bboxes.shape[0]):
        c = bboxes[i, :]
        fl = (c[0, 0] * 10 + 400, -c[1, 0] * 10 + 400)
        fr = (c[0, 1] * 10 + 400, -c[1, 1] * 10 + 400)
        bl = (c[0, 2] * 10 + 400, -c[1, 2] * 10 + 400)
        br = (c[0, 3] * 10 + 400, -c[1, 3] * 10 + 400)
        
        if categories[i].item() == 2 or categories[i].item() == 5 or categories[i].item() == 4:
            drawer.polygon([fl, fr, br, bl], fill=(255, 255, 255))
    
    return image2

def crop_by_view(masked_bbox_image_PIL, view, size):
    image2 = masked_bbox_image_PIL.copy()
    a = size[0]/2
    if view == 'front_left':
        image2 = image2.crop((a, 0, 2 * a, a))
    elif view == 'front_right':
        image2 = image2.crop((a, a, 2 * a, 2 * a))
    elif view == 'back_left':
        image2 = image2.crop((0, 0, a, a))
    elif view == 'back_right':
        image2 = image2.crop((0, a, a, 2 * a))
    elif view == 'front':
        image2 = image2.crop((a, a / 2, 2 * a, 3 * a / 2))
        image2 = image2.rotate(90, expand = True)
    elif view == 'back':
        image2 = image2.rotate(270, expand = True)
        image2 = image2.crop((a / 2, 0, 3 * a / 2, a))
        
    return image2


# parser = OptionParser()
# parser.add_option('--train-views', dest = 'train_views', default = 0, type = 'string', help = 'views to be trained (comma separated)')
# (options, args) = parser.parse_args()

# views = options.train_views.replace(' ', '').split(',')

views = ['front_left', 'front', 'front_right', 'back_left', 'back', 'back_right']

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = '/home/prs392/codes/top_view_predictor/artifacts/data'

image_folder = data_dir
annotation_csv = f'{data_dir}/annotation.csv'
    

batch_size = 32

train_pre_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

val_pre_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_post_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_post_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_set = LabeledDataset(
    image_folder = image_folder,
    scene_index = np.arange(106, 131),
    annotation_file=annotation_csv,
    transform = train_pre_transforms
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

val_set = LabeledDataset(
    image_folder = image_folder,
    scene_index = np.arange(131, 134),
    annotation_file=annotation_csv,
    transform = val_pre_transforms
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

directory = '/scratch/prs392/data/submission_monolayout'

view_to_index = {
    'front_left': 0,
    'front': 1,
    'front_right': 2,
    'back_left': 3,
    'back': 4,
    'back_right': 5
}

count = 0

for sample, target, road_image, extra in train_loader:
    
    assert len(sample) == len(road_image)
    
    for i in range(len(road_image)):
        
        file_name = "{0:0=6d}".format(count) + ".png"
        
        road_image_tensor = road_image[i].int()
        road_image_tensor[road_image_tensor == 1] = 255
        road_image_PIL = torchvision.transforms.ToPILImage(mode = None)(road_image_tensor.cpu()).convert('RGB')
        blank_top_view = Image.new(mode='RGB', size = (road_image_PIL.size), color=(0,0,0))
        bbox_image_PIL = draw_bboxes(blank_top_view, target[i]['bounding_box'], target[i]['category'])
        bbox_image_PIL.save(f"{directory}/merged_top_views/{file_name}")

        for view in views:
            
            original = torchvision.transforms.ToPILImage(mode='RGB')(sample[i][view_to_index[view]]).resize((512, 512))
            original.save(f"{directory}/{view}/image_2/{file_name}")
            
            bbox_image_PIL_2 = bbox_image_PIL.copy()

            masked_bbox_image_PIL = mask_bbox_image_by_view(bbox_image_PIL_2, view, (800, 800))
            
            masked_bbox_image_tensor = to_tensor(masked_bbox_image_PIL)
            
            test = torchvision.transforms.ToPILImage(mode = 'RGB')(masked_bbox_image_tensor.cpu())
            
            # Cropping
            
            cropped_image = crop_by_view(masked_bbox_image_PIL, view, (800, 800)).resize((256, 256))
            
            cropped_image.save(f"{directory}/{view}/TV_car/{file_name}")
            
        count += 1
            
for sample, target, road_image, extra in val_loader:
    
    assert len(sample) == len(road_image)
    
    for i in range(len(road_image)):
        
        file_name = "{0:0=6d}".format(count) + ".png"
        
        road_image_tensor = road_image[i].int()
        road_image_tensor[road_image_tensor == 1] = 255
        road_image_PIL = torchvision.transforms.ToPILImage(mode = None)(road_image_tensor.cpu()).convert('RGB')
        blank_top_view = Image.new(mode='RGB', size = (road_image_PIL.size), color=(0,0,0))
        bbox_image_PIL = draw_bboxes(blank_top_view, target[i]['bounding_box'], target[i]['category'])
        bbox_image_PIL.save(f"{directory}/merged_top_views/{file_name}")

        for view in views:
            
            original = torchvision.transforms.ToPILImage(mode='RGB')(sample[i][view_to_index[view]]).resize((512, 512))
            original.save(f"{directory}/{view}/image_2/{file_name}")
            
            bbox_image_PIL_2 = bbox_image_PIL.copy()

            masked_bbox_image_PIL = mask_bbox_image_by_view(bbox_image_PIL_2, view, (800, 800))
            
            masked_bbox_image_tensor = to_tensor(masked_bbox_image_PIL)
            
            test = torchvision.transforms.ToPILImage(mode = 'RGB')(masked_bbox_image_tensor.cpu())
            
            # Cropping
            
            cropped_image = crop_by_view(masked_bbox_image_PIL, view, (800, 800)).resize((256, 256))
            
            cropped_image.save(f"{directory}/{view}/TV_car/{file_name}")
            
        count += 1