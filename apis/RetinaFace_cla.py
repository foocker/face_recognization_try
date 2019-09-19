from RF.inference import get_model, img_tensor,tensor_img, inference
import math
from PIL import Image


class Retinafce_Detect():
    def __init__(self, model_path, score_threshold=0.8, iou_threshold=0.5, thre_cos=15):
        self.model_path = model_path
        self.score_thre = score_threshold
        self.iou_thre = iou_threshold
        self.thre_cos = thre_cos
        self.RetinaFace = get_model(self.model_path)

    def infer_video(self, frame):
        img_t= img_tensor(frame)
        picked_boxes, picked_landmarks = inference(img_t, self.RetinaFace, self.score_thre, self.iou_thre)
        img_np = tensor_img(img_t)
        bboxes, landmarks, faces = [], [], []
        if len(picked_boxes[0]) is None:
            return bboxes, landmarks
        else:
            for j, boxes in enumerate(picked_boxes):
                for box,landmark in zip(boxes,picked_landmarks[j]):
                    box = box.cpu().numpy().astype(int)
                    landmark = landmark.cpu().numpy().astype(int)
                    flag = self.quality_coords(box, landmark, thre_cos=self.thre_cos)
                    if not flag:
                        continue
                    bboxes.append(box)
                    landmarks.append(landmark)
                    faces.append(Image.fromarray(frame[box[1]:box[3],box[0]:box[2]]))
        return bboxes, landmarks, faces

    def infer_dir(self, img_dir):
        pass

    def quality_coords(self, bbox, coords, thre_cos=15):
        # th_cos small will converge -1 :decrease
        x1, y1, x2, y2 = bbox
        xs = coords[0::2]
        ys = coords[1::2]
        # print("????", x1, y1, x2, y2)
        coordx_min , coordy_min= min(xs), min(ys)
        coordx_max, coordy_max = max(xs), max(ys)
        # print("xsys", xs, ys)
        flagl = coordx_min > x1 and coordy_min > y1
        flagr = coordx_max < x2 and coordy_max < y2
        flaglocat = flagl and flagr
        vec31 = (coords[0] - coords[4], coords[1] - coords[5])
        vec34 = (coords[4] - coords[6], coords[5] - coords[7])    # 43
        vec32 = (coords[2] - coords[4], coords[3] - coords[5])
        vec35 = (coords[4] - coords[8], coords[5] - coords[9])    # 45
        cos134 = (vec31[0]*vec34[0] + vec31[1]*vec34[1]) / math.sqrt((vec31[0]**2 + vec31[1]**2) * (vec34[0]**2+vec34[1]**2))
        cos235 = (vec32[0]*vec35[0] + vec32[1]*vec35[1]) / math.sqrt((vec32[0]**2 + vec32[1]**2) * (vec35[0]**2+vec35[1]**2))
        flagcos = math.cos(3.1415/180*thre_cos) > max(abs(cos134), abs(cos235))
        print("+++]]]]]]]++++", flaglocat, cos134, cos235, flagcos)

        return flaglocat and flagcos