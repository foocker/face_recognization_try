from apis.faceboxes_detect_cla import FaceBoxesDetect
from configs import cfg


FBD = FaceBoxesDetect(cfg)
FBD.prepare()
FBD.test_video()
