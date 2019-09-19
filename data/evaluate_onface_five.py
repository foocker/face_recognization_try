from apis.faceboxes_detect_cla import FaceBoxesDetect
from configs.config_detect import cfg
import cv2


print("loading faceboxes model!!")
faceboxes = FaceBoxesDetect(cfg)
faceboxes.prepare()
print('faceboxes loaded!!\n')


# coords = list(map(int, info['coords']))
# img = cv2.imread(os.path.join('./data/alfw_rect_coords/flickr', info['filename']))
# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
# polylines = []
# for i in range(0, 10, 2):
#     polylines.append([coords[i], coords[i+1]])
#     cv2.circle(img, (coords[i], coords[i+1]), 5, (0, 255, 0), -1)
# cv2.polylines(img, [np.array(polylines, np.int32)],True, (0, 255, 255), 4)