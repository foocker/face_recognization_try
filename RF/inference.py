import cv2
from RF.eval_widerface import get_detections
from RF.torchvision_model import create_retinaface
import torch
import numpy as np


def get_model(model_path):
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()
    return RetinaFace


def img_tensor(img):
    # img = cv2.imread(file)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    return img


def inference(img, retinaface, score_threshold, iou_threshold):
    picked_boxes, picked_landmarks = get_detections(img, retinaface, score_threshold, iou_threshold)
    return picked_boxes, picked_landmarks


def show_infer(img, picked_boxes, picked_landmarks):
    picked_boxes,picked_landmarks = np.ceil(picked_boxes[0].cpu().numpy()), np.ceil(picked_landmarks[0].cpu().numpy())
    for bbox, landmark in zip(picked_boxes, picked_landmarks):
        img = draw_box(img, bbox)
        img = draw_ploylines(img, landmark)
    cv2.imwrite('result.jpg', img)
    cv2.imshow('pred', img)
    return img


def tensor_img(img_t):
    np_img = img_t.cpu().squeeze(0).permute(1, 2, 0).numpy()
    np_img.astype(int)
    img = np_img.astype(np.uint8)
    return img

def draw_box(img, bbox):
    bbox = list(map(int, bbox))
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
    return img


def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame


def draw_ploylines(img, coords):
    coords = list(map(int, coords))
    polylines = []
    for i in range(0, 10, 2):
        polylines.append([coords[i], coords[i + 1]])
        img = cv2.circle(img, (coords[i], coords[i + 1]), 5, (0, 255, 0), -1)
        img = cv2.polylines(img, [np.array(polylines, np.int32)], True, (0, 255, 255), 4)
    return img


