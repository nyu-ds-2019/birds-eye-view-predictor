

import os
from shutil import copyfile

camera_dir = '/home/prs392/codes/top_view_predictor/artifacts/data'
image_dir = '/home/ns4486/birds_eye_view_app_dir/artifacts/data/yolo_data/images'

scenes = list(range(120, 134))
train_scenes = list(range(120, 131))
val_scenes = list(range(131, 134))
samples = list(range(126))



for scene in scenes:
    for sample in samples:
        src_dir = os.path.join(camera_dir, f'scene_{scene}', f'sample_{sample}')
        
        for image_name in ['CAM_BACK_LEFT.jpeg', 'CAM_BACK.jpeg', 'CAM_BACK_RIGHT.jpeg', 'CAM_FRONT_LEFT.jpeg', 'CAM_FRONT.jpeg', 'CAM_FRONT_RIGHT.jpeg']:
            copyfile(
                os.path.join(src_dir, image_name), 
                os.path.join(image_dir, f'scene_{scene}_sample_{sample}_{image_name}')
            )
    
    print('completed scene ' + str(scene))


for scene in train_scenes:
    for sample in samples:
        for image_name in ['CAM_BACK_LEFT.jpeg', 'CAM_BACK.jpeg', 'CAM_BACK_RIGHT.jpeg', 'CAM_FRONT_LEFT.jpeg', 'CAM_FRONT.jpeg', 'CAM_FRONT_RIGHT.jpeg']:
            path = os.path.join(image_dir,  f'scene_{scene}_sample_{sample}_{image_name}')
            if os.path.exists(path):
                print(path)

df = pd.read_csv('/home/prs392/codes/top_view_predictor/artifacts/data/annotation.csv')
df['c_x'] = (df['bl_x'] + df['fr_x']) / 2
df['c_y'] = (df['bl_y'] + df['fr_y']) / 2
df['w'] = np.sqrt((df['fl_x'] - df['bl_x']) ** 2 + (df['fl_y'] - df['bl_y']) ** 2)
df['h'] = np.sqrt((df['fl_x'] - df['fr_x']) ** 2 + (df['fl_y'] - df['fr_y']) ** 2)


























from math import degrees
from math import atan2

a = 400

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


file1 = open('/home/ns4486/birds_eye_view_app_dir/artifacts/data/yolo_data/yolo_targets_1.csv', 'r') 
Lines = file1.readlines() 

count = 0
# Strips the newline character 
for line in Lines: 
    if count == 0:
        continue
    
    line = line.split(' ')
    line.pop(0)
    
    scene = line[0]
    sample = line[1]
    class_id = line[2]
    c_x = line[3]
    c_y = line[4]
    w = line[5]
    h = line[6]

    view = None
    
    if in_front(c_x * 800 - a, -c_y * 800 + a):
        view = 'FRONT'
    if in_front_left(c_x * 800 - a, -c_y * 800 + a):
        view = 'FRONT_LEFT'
    if in_front_right(c_x * 800 - a, -c_y * 800 + a):
        view = 'FRONT_RIGHT'
    if in_back(c_x * 800 - a, -c_y * 800 + a):
        view = 'BACK'
    if in_back_left(c_x * 800 - a, -c_y * 800 + a):
        view = 'BACK_LEFT'
    if in_back_right(c_x * 800 - a, -c_y * 800 + a):
        view = 'BACK_RIGHT'
    
    if view == 'FRONT':
        file2 = open(f'/home/ns4486/birds_eye_view_app_dir/artifacts/data/yolo_data/labels/scene_{scene}_sample_{sample}_CAM_{view}.txt', 'a+')
        file2.write(str(class_id) + ' ' + str(c_x) + ' ' + str(c_y) + ' ' + str(w) + ' ' + str(h))
        file2.close()
    
    count += 1
    