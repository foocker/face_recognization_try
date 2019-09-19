import cv2
import os
import numpy as np


def pred_imgcoords(data_root, data_name, txt_file):
    with open(os.path.join(data_root, txt_file), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line[:-1]
            line = line.split(' ')
            line = line[1:]
            file_path = line[0]
            img = cv2.imread(os.path.join( data_root, data_name,file_path))
            coords = list(map(float, line[1:11]))
            coords = list(map(int, coords))
            polylines = []
            for i in range(0, 10, 2):
                polylines.append([coords[i], coords[i+1]])
                cv2.circle(img, (coords[i], coords[i+1]), 5, (0, 255, 0), -1)
            cv2.polylines(img, [np.array(polylines, np.int32)],True, (0, 255, 255), 4)
            cv2.imwrite(os.path.join('./data/face_five/pred_result/{}.jpg'.format(file_path.split('/')[-1])), img)


# data = np.loadtxt("data.txt") 

pred_imgcoords('./data/face_five', 'AFLW', 'testing.txt')