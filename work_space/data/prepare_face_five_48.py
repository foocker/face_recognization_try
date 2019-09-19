import os
import cv2
import numpy as np


def faces_five_48(data_dir, txt_file):
    data_info = []
    with open(os.path.join(data_dir, txt_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            f_dir, f_name  = line[0].split('\\')
            x1, x2, y1, y2 = list(map(int, line[1:5]))
            # x2, y2 = x1 + w, y1 + h
            points = np.array(list(map(float, line[5:])))

            img = cv2.imread(os.path.join(data_dir, f_dir, f_name))
            if y1<y2 and x1<x2:
                img_crop = img[y1:y2, x1:x2]
            else:
                continue
            w_c, h_c = x2 - x1, y2 - y1
            relat = np.tile([x1, y1], 5)
            points_crop = points - relat
            img_resized, points_resized = resize_reloc(img_crop, points_crop)
            new_dir = os.path.join(data_dir,f_dir+'_crop_test')
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            img_info = {'filepath':os.path.join(new_dir, f_name),
                    'points':points_resized}
            data_info.append(img_info)
            cv2.imwrite(os.path.join(new_dir, f_name), img_resized)

        np.save('./data/face_five/face_five_crop_48_test.npy', data_info)


def resize_reloc(img, points, size=48):
    # let max lenght is 48 and pad to square
    h, w, _ = img.shape
    max_ = max(h, w)
    if max_ == 0:
        print(img)
    ratio = size / max_
    n_h, n_w = int(ratio * h+0.5), int(ratio * w+0.5)
    assert n_h > 0 and n_w >0
    assert n_h == size or n_w == size
    img_resized = cv2.resize(img, (n_w, n_h))
    img_container = np.zeros((size, size, 3), dtype=np.uint8)
    points *= ratio
    if n_h == 48:
        left = 48 - n_w
        print(img_resized[:,:,0].shape, left)
        img_container[:,:,0] =  np.pad(img_resized[:,:,0], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))  
        img_container[:,:,1] =  np.pad(img_resized[:,:,1], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,2] =  np.pad(img_resized[:,:,2], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))
        points[0::2] += left
    else:
        top = 48 - n_h   
        print(img_resized[:,:,0].shape, top)
        img_container[:,:,0] =  np.pad(img_resized[:,:,0], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,1] =  np.pad(img_resized[:,:,1], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,2] =  np.pad(img_resized[:,:,2], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        points[1::2] += top
    return img_container, points  


def test_crop(data_info):
    info = np.load(data_info)
    ex = info[0]
    img = cv2.imread(ex['filepath'])
    points = list(map(np.ceil, ex['points']))
    points = list(map(int, points))
    for i in range(0, 10, 2):
        cv2.circle(img, (points[i], points[i+1]), 5, (0, 255, 0), -1)
    cv2.imwrite('./img_g.jpg', img)
    for i in info:
        im = cv2.imread(i['filepath'])
        h, w, _ = im.shape
        if h != w:
            print(h, w, i['filepath'])


#faces_five_48('./data/face_five', 'trainImageList.txt')
# faces_five_48('./data/face_five', 'testImageList.txt')

#test_crop('./data/face_five/face_five_crop_48.npy')


def recrop_detect(detetc_face, size=48):
    h, w, _ = detetc_face.shape
    max_ = max(h, w)
    if max_ == 0:
        print(detetc_face)
    ratio = size / max_
    n_h, n_w = int(ratio * h+0.5), int(ratio * w+0.5)
    assert n_h > 0 and n_w >0
    assert n_h == size or n_w == size
    img_resized = cv2.resize(detetc_face, (n_w, n_h))
    img_container = np.zeros((size, size, 3), dtype=np.uint8)
    if n_h == size:
        left = size - n_w
        img_container[:,:,0] =  np.pad(img_resized[:,:,0], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))  
        img_container[:,:,1] =  np.pad(img_resized[:,:,1], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,2] =  np.pad(img_resized[:,:,2], ((0, 0), (left, 0)), 'constant', constant_values=(0, 0))
    else:
        top = size - n_h   
        img_container[:,:,0] =  np.pad(img_resized[:,:,0], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,1] =  np.pad(img_resized[:,:,1], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        img_container[:,:,2] =  np.pad(img_resized[:,:,2], ((top, 0), (0, 0)), 'constant', constant_values=(0, 0))
        
    return img_container