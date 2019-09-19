import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

	    # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def _crop_(image, boxes, coords, labels, img_dim):
    # bboxes, [n, 4],x1, y1, x2, y2
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)    # ��scale��С��С�ߣ���������
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)    # ��ʣ���������ѡ��ƽ�ƾ���
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)    # �߶�����
        roi = np.array((l, t, l + w, t + h))    # ��ȡƽ�ƺ��roi:(l, t),(��)
        roi_coords = np.array((l, t, l+w, t, l+int(w/2), t+int(h/2), l, t+h, l+w, t+h))
        

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)    # crop box contain max(bbox)?
        if not flag.any():
            continue
        # print("roi:", roi, "bbox", boxes)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2    # ��ʵbbox������
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)  # all roi containe bbox center, is bool
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()    # its correspond box label
        coords_t = coords[mask_a].copy()    # ѡȡroi����bbox���ĵ�bbox, ����Ӧ��label, coord
        # print("mask_a",mask_a,"boxes_t", boxes_t, "\n","labels_t", labels_t)

        if boxes_t.shape[0] == 0:   # means no box in roi
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]    # img is h, w
        # print("before+++bbox", boxes_t, "====roi", roi, "||||roi_coords", roi_coords)
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])    # return bigger, means inner coordinate
        
        boxes_t[:, :2] -= roi[:2]     # bbox ��roi����Ծ��뱣��һ��(����img�ѱ���ȡ)
        # coords[:, ]   ?????????????????????????
        
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])    # return smaller, means inner coordinate
        boxes_t[:, 2:] -= roi[:2]    # ֻ�����ƶ�!!!!!!
        # print("after+++++bbox", boxes_t)
        # print("relative value",boxes_t)
        # coords[:, ]   ?????????????????????????
        # print("before+++coords:", coords )
        coords_t -= np.tile(roi[:2], 5)
        # print("after+++++coords",coords_t)
        ## coords 
        """
        coords_t[:, :2] = np.maximum(coords_t[:, :2], roi_coords[:2])  
        coords_t[:,:2]  -= roi_coords[:2]
        coords_t[:, 2:4] = np.maximum(coords_t[:, 2:4], roi_coords[2:4])
        coords_t[:,:2]  -= roi_coords[2:4]
        ## coords center ###
        
        left_rigt_center = coords_t[:, 4] - roi_coords[4]    #w or h
        for flag in left_rigt_center:
            if flag: 
                coords_t[:, 4:6] = np.maximum(coords_t[:, 4:6], roi_coords[4:6])
                coords_t[:,4:6]  -= roi_coords[4:6]
            else:
                coords_t[:, 4:6] = np.minimum(coords_t[:, 4:6], roi_coords[4:6])
                coords_t[:,4:6]  -= roi_coords[4:6]
        
        ## coords center ###
        coords_t[:, 6:8] = np.minimum(coords_t[:, 6:8], roi_coords[6:8])
        coords_t[:,6:8]  -= roi_coords[6:8]
        coords_t[:, 8:] = np.minimum(coords_t[:, 8:], roi_coords[8:])
        coords_t[:,8:]  -= roi_coords[8:]
        """
        # coords

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim    # 1024=img_dim scale
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 16.0    # why in img_dim??
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        coords_t = coords_t[mask_b]

        if boxes_t.shape[0] == 0:    # no box satisfy above condition
            continue

        pad_image_flag = False

        return image_t, boxes_t, coords_t, labels_t, pad_image_flag
    return image, boxes, coords, labels, pad_image_flag

def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes
    
def _mirror_(image, boxes, coords):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        coords = coords.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]    # x1, x2 = w -x2, w-x1
        coords[:, 0::2] = width - coords[:, 0::2][::-1]    # or coords[:, 0::2][::-1]
    return image, boxes, coords


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t = _mirror(image_t, boxes_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t


class preproc_(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        # print(targets, targets.shape)
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()    # change label
        coords = targets[:, 5:].copy()

        image_t, boxes_t, coords_t, labels_t, pad_image_flag = _crop_(image, boxes, coords, labels, self.img_dim)    # change
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t, coords_t = _mirror_(image_t, boxes_t, coords_t)    # change
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        coords_t[:, 0::2] /= width
        coords_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t,coords_t))

        return image_t, targets_t
