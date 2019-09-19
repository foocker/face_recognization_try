import os
import cv2
import torch
import numpy as np
from configs.config_detect import cfg
from utils.load_arc import load_model
from utils.nms_wrapper import nms
from utils.box_utils import decode
import torch.backends.cudnn as cudnn
from models.faceboxes import FaceBoxes
from layers.functions.prior_box import PriorBox


def getinfo(root):
    """
    root/1
    root/2
    ...
    :param root:
    :return:{"root/1":[img1, img2, ...], ...}
    """
    pathinfo = {}
    for dir in os.listdir(root):
        dic = os.path.join(root, dir)
        imgs = os.listdir(dic)
        pathinfo.update({dic: imgs})
    return pathinfo


def img_crop_face(root_, pathinfo, mode="train", confidence_threshold=0.05, top_k=10):
    # move rate img from train set to val set
    # root = "/aidata/dataset/faces/CASIA-FaceV5/train"
    # root_ = "/aidata/dataset/faces/CASIA-FaceV5_Crop"
    # to 112*112 for arc training
    resize=1
    if not os.path.exists(root_):
        os.mkdir(root_)
    if not os.path.exists(root_ + "/" + mode):
        os.mkdir(root_ + "/" + mode)
    for key, imgs in pathinfo.items():
        newdir = os.path.join(root_, "train", os.path.split(key)[1])
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        for img_p in imgs:
            img_raw = cv2.imread(os.path.join(key, img_p), cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            loc, conf = net(img)
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, 0.3, force_cpu=False)    # nms_threshold
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:5, :]    # keep_top_k

            for inx, b in enumerate(dets):
                if b[4] < 0.7:
                    continue
                # text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                crop_img = img_raw[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]
                crop_resized = cv2.resize(crop_img, (112, 112))
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                # cx = b[0]
                # cy = b[1] + 12
                # cv2.putText(img_raw, text, (cx, cy),
                #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imwrite(os.path.join(newdir, img_p, str(inx), '.jpg'), crop_img)


if __name__ == '__main__':
    trained_model = ''
    load_to_cpu = False
    torch.set_grad_enabled(False)
    net = FaceBoxes(phase='test', size=None, num_classes=2)
    net = load_model(net, trained_model, load_to_cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    root = "/aidata/dataset/faces/CASIA-FaceV5/train"
    root_ = "/aidata/dataset/faces/CASIA-FaceV5_Crop"
    pathinfo = getinfo(root)

    img_crop_face(root_, pathinfo, mode="train")




