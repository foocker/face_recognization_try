from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from configs import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from PIL import Image
from utils.load_arc import load_model
from models import ONet
from data.prepare_face_five_48 import recrop_detect
from apis.ttOnet import load_onet, points_infer, filter_face


class FaceBoxesDetect():
    def __init__(self, cfg):
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.Onet = None
        
    def prepare(self):
        self.model = load_model(self.net, self.cfg['trained_model'], False)
        self.model.eval()
        cudnn.benchmark = True
        self.model = self.model.to(self.device)
        print('Finished loading detect model!')
        self.Onet = load_onet(ONet(),self.cfg['Onet_weights'], self.device)
        # self.Onet.eval()
        print('Finished loading Five Points model!')
    
    def facebox_detect(self, img_raw):
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
    
        loc, conf = self.model(img)  # forward pass
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]
    
        # ignore low scores
        inds = np.where(scores > self.cfg['confidence_threshold'])[0]
        boxes = boxes[inds]
        scores = scores[inds]
    
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.cfg['top_k']]
        boxes = boxes[order]
        scores = scores[order]
    
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.cfg['nms_threshold'],False)
        dets = dets[keep, :]
    
        # keep top-K faster NMS
        boxes_score = dets[:self.cfg['keep_top_k'], :]
        # boxes_score[:, :-1] += 1
        # remove the locat is not positive
        po_ng = np.array([np.any(box<0) for box in boxes_score])
        boxes_score = boxes_score[np.where(po_ng==False)]
    
        return boxes_score
    
    def align_multi(self, img_ori, boxes_score, save=True, limit=None, obj_thr=0.7):
        # [[x1,y1,x2,y2,score],[...]]
        h, w, _ = img_ori.shape
        if limit:
            boxes_score = boxes_score[:limit]
        faces = []
        fiter_bboxes = []
        shoulders = []
        for inx, b in enumerate(boxes_score):
            # do some filter func
            if b[4] < obj_thr:
                continue
            shoulder_box =  self.shoulder_face(h, w, b)
            shoulders.append(shoulder_box)
            score = b[-1]
            b = list(map(int, b))
            score_bb = shoulder_box[-1]
            bb = list(map(int, shoulder_box))
            
            flag = False
            try:
                crop_img = img_ori[b[1]:b[3], b[0]:b[2]]    # some location is negative
                # crop_img_bb = img_ori[bb[1]:bb[3], bb[0]:bb[2]]
                # crop_resized = cv2.resize(crop_img, (112, 112))
                flag = True
            except:
                print(img_ori,img_ori.shape, (b[1],b[3],b[0],b[2]))
                # raise "crop error"
                continue
            recrop_img = recrop_detect(crop_img, size=48)
            points = points_infer(self.Onet, recrop_img, self.device)
            k, scale = filter_face(points)
            if scale < 0.15 or scale > 8:
                flag = False
                continue
            if save:
                for i in range(0, 10, 2):
                    cv2.circle(recrop_img, (points[i], points[i+1]), 5, (0, 255, 0), -1)
                cv2.imwrite('./data/eval48/{}_{}.jpg'.format(np.random.randint(1000), inx), recrop_img)
            crop_resized = cv2.resize(crop_img, (112, 112))
            if flag:
                faces.append(Image.fromarray(crop_resized))
                b[-1] = score
                fiter_bboxes.append(b)
            if not save:
                cv2.imwrite('./data/eval/{}_{}.jpg'.format(np.random.randint(1000), inx), crop_resized)
                # cv2.imwrite('./data/eval/{}_{}.jpg'.format(np.random.randint(1000), inx), crop_img_bb)
        return np.array(fiter_bboxes), np.array(shoulders), faces
    
    def draw_box_score(self, frame, boxes_score):
        for b in boxes_score:
            for b in boxes_score:
                # cv2.rectangle need int, not np.float
                text = "{:.2f}".format(b[4])
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                cx = int(b[0])
                cy = int(b[1]) + 12
                cv2.putText(frame, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return frame
        
    def test_video(self):
        cap = cv2.VideoCapture(self.cfg['v_path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps:", fps, type(fps))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(self.cfg['sv_path'], cv2.VideoWriter_fourcc(*'XVID'), int(fps), size)
    
        while cap.isOpened():
            isSuccess, frame = cap.read()
            if isSuccess:
                dets = self.facebox_detect(frame)
                if dets.shape[0] !=0:
                    boxes_score, _, faces = self.align_multi(frame, dets, save=True)
                    # print(boxes_score.shape, len(faces), boxes_score)
                    
                    if len(faces)!= 0:
                        frame = self.draw_box_score(frame, boxes_score)
                    
                        
                video_writer.write(frame)
                
        cap.release()
        video_writer.release()
        
    def test_img(self, img_dir):
        pass
        
    def shoulder_face(self, h, w, bbox_s):
        # h, w, _ = img.shape
        sca = np.random.random() + 2
        rat = np.random.uniform(1, 1.3)
        x1, y1, x2, y2 = bbox_s[:-1]
        w_f, h_f = x2 - x1, y2 - y1
        expand_w = h_f * sca
        expand_h =  h_f / w_f * expand_w
        x1_e = x1 - 0.5*(expand_w-w_f) if x1 - 0.5*(expand_w-w_f) > 0 else 1
        x2_e = x2 + 0.5*(expand_w-w_f) if x2 + 0.5*(expand_w-w_f) < w else w-1
        y1_e = y1 - 0.25*(expand_h-h_f) if y1 - 0.25*(expand_h-h_f)> 0 else 1
        y2_e = y1 + 0.75*(expand_h-h_f) if y2 + 0.75*(expand_h-h_f) < h else h-1
        
        return [x1_e, y1_e, x2_e, y2_e, bbox_s[-1]]
        

    def face_shoulder(self, img_dir):
        sca = np.random.random() + 2
        rat = np.random.uniform(1, 1.3)
        face_h_w, shoulder_w_h, scale, equ = [], [], [], []
        for img_n in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, img_n))
            h, w, _ = img.shape
            dets = self.facebox_detect(img)
            if dets.shape[0] != 0:
                
                x1, y1, x2, y2 = dets[0][:-1]
                w_f, h_f = x2 - x1, y2 - y1
                [x1_e, y1_e, x2_e, y2_e] = self.shoulder_face(h,w, dets[0])
                we, he = x2_e - x1_e, y2_e - y1_e
                face_h_w.append(h_f/w_f)
                shoulder_w_h.append(w/h)
                scale.append((h/w_f, w/h_f))
                equ.append((we/w, he/h))
            else:
                print(img_n, "dets is: ",dets)
        return face_h_w, shoulder_w_h, scale, equ


if __name__ == '__main__':
    FBD = FaceBoxesDetect(cfg)
    FBD.prepare()
    FBD.test_video()
    # face_h_w, shoulder_w_h, scale, equ = FBD.face_shoulder('./data/face_shoulder/')
    # print("face_h_w:{ },\n, shoulder_w_h:{}, \n, scale:{}, \n, equ:{}".format(face_h_w, shoulder_w_h, scale, equ))

