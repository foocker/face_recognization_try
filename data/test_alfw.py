import numpy as np
import cv2
import os

from data import AFLW, preproc_


def show_img_info(img_info):
    all_info = np.load(img_info)
    print(all_info[1:4])
    if not os.path.exists('./data/rect_coords_draw'):
        os.mkdir('./data/rect_coords_draw')
    c = 0
    for info in  all_info:
        x1, y1, x2, y2 = list(map(int, info['bbox']))
        # [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5] = list(map(int, info['coords']))
        coords = list(map(int, info['coords']))
        img = cv2.imread(os.path.join('./data/alfw_rect_coords/flickr', info['filename']))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        polylines = []
        for i in range(0, 10, 2):
            polylines.append([coords[i], coords[i+1]])
            cv2.circle(img, (coords[i], coords[i+1]), 5, (0, 255, 0), -1)
        cv2.polylines(img, [np.array(polylines, np.int32)],True, (0, 255, 255), 4)
    
        # cv2.imwrite(os.path.join('./data/rect_coords_draw', info['filename'][2:]), img)
    # cv2.imshow()


# show_img_info('./data/all_info_.npy')
img_dim = 1024
rgb_mean = (104, 117, 123) 
npy_file = './data/all_info_.npy'
training_dataset = './data/alfw_rect_coords/flickr'
dataset = AFLW(training_dataset, npy_file, preproc_(img_dim, rgb_mean))


def show_map_bbox_coords(num):
    for i in range(num):
        item = np.random.randint(10000)
        img, target = dataset.__getitem__(item)
        im = img.numpy().transpose(1, 2, 0)
        t = target*img.shape[1]
        t[0, 4] /= img.shape[1]
        x1, y1, x2, y2 = int(t[0][0]), int(t[0][1]), int(t[0][2]), int(t[0][3])
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
        coords = list(map(int, t[0][5:]))
        polylines = []
        for j in range(0, 10, 2):
            polylines.append([coords[j], coords[j+1]])
            cv2.circle(im, (coords[j], coords[j+1]), 5, (0, 255, 0), -1)
            cv2.polylines(im, [np.array(polylines, np.int32)], True, (0, 255, 255), 4)
        cv2.imwrite('imgmap_bbox_coords_{}.jpg'.format(i), im)
        
# show_map_bbox_coords(200)
#  scp -r root@172.16.0.208:/root/Codes/FaceBoxes_Arc/imgmap_bbox_coords_*.jpg /e/Face_map_bbox_coords


def face_quality():
    pass
    
    
    
    
    