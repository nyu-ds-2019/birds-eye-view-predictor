import torch
import torchvision
from torchvision import transforms
from PIL import ImageDraw
from math import tan, radians


def shift_coordinates(x, y, a):
    return (x + a, -y + a)

def get_masking_polygon_by_view(view, a):
    a = a / 2
    polygon_map_centered_axes = {
        'front_left': [
            (0, 0),
            (a/tan(radians(95)) + a, 0),
            (a, a),
            (2 * a, -a * tan(radians(25)) + a),
            (2 * a, 2 * a),
            (0, 2 * a)
        ],

        'front': [
            (0, 0),
            (2 * a, 0),
            (2 * a, -a * tan(radians(35)) + a),
            (a, a),
            (2 * a, a * tan(radians(35)) + a),
            (2 * a, 2 * a),
            (0, 2 * a)
        ],

        'front_right': [
            (0, 0),
            (2 * a, 0),
            (2 * a, a * tan(radians(25)) + a),
            (a, a),
            (a - a/tan(radians(85)), 2 * a),
            (0, 2 * a)
        ],

        'back_left': [
            (a + a/tan(radians(85)), 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (0, 2 * a),
            (0, -a * tan(radians(25)) + a),
            (a, a)
        ],

        'back': [
            (0, 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (0, 2 * a),
            (0, a * tan(radians(35)) + a),
            (a, a),
            (0, -a * tan(radians(35)) + a)
        ],

        'back_right': [
            (0, a * tan(radians(25)) + a),
            (0, 0),
            (2 * a, 0),
            (2 * a, 2 * a),
            (a - a / tan(radians(95)), 2 * a),
            (a, a)
        ]
    }

    return polygon_map_centered_axes[view]

'''
masks the rest of the camera views in the top view

road_image = PIL image
view = front_left / front / front_right / back_left / back / back_right
size = (h, h) of the road image
'''

normalize_transform = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)


def mask_road_image_by_view(road_image_batched, view, size):
    batch_size = len(road_image_batched)
    road_image_list = []

    for batch_num in range(batch_size):
        road_image = road_image_batched[batch_num].int()
        road_image[road_image == 1] = 255
        road_image_PIL = to_PIL(road_image).convert('RGB')
        polygon_coordinates = get_masking_polygon_by_view(view, size[0])
        road_image_masked = road_image_PIL.copy()

        drawer = ImageDraw.Draw(road_image_masked)
        drawer.polygon(polygon_coordinates, fill = (100, 100, 100))
        road_image_masked = road_image_masked.resize((128, 128))
        road_image_masked_tensor = to_tensor(road_image_masked)
        road_image_masked_tensor = normalize_transform(road_image_masked_tensor)
        road_image_masked_tensor = road_image_masked_tensor.view([1, *road_image_masked_tensor.shape])

        road_image_list.append(road_image_masked_tensor)

    return torch.stack(road_image_list, dim = 0)


def to_PIL(image_tensor):
    return torchvision.transforms.ToPILImage(mode = None)(image_tensor.cpu())

def to_tensor(PIL_image):
    return torchvision.transforms.functional.to_tensor(PIL_image)
