import cv2
import torch
from configs.config_arc import get_config
from apis.learner import face_learner
from utils.utils_arc import load_facebank, draw_box_name, prepare_facebank
import numpy as np
from torchvision import transforms as trans
from models.arcface import l2_norm
from configs.config_detect import cfg

from datetime import datetime
import os

from apis.faceboxes_detect_cla import FaceBoxesDetect
from apis.RetinaFace_cla import Retinafce_Detect


def new_person(faces, path_new_face):
    for face in faces:
        cv2.imwrite(
            os.path.join(path_new_face, '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "_").replace(" ", "_"))),
            face)


def add_new_emd(conf, model, faces, tta=True):
    model.eval()
    new_name = 'new_0'
    embs = []
    names = []
    for i, face in enumerate(faces):
        emb = model(conf.test_transform(face).to(conf.device).unsqueeze(0))
        names.append(new_name[:4] + str(int(new_name[4:]) + i))
        if tta:
            mirror = trans.functional.hflip(face)
            emb = model(conf.test_transform(face).to(conf.device).unsqueeze(0))
            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:
            embs.append(model(conf.test_transform(face).to(conf.device).unsqueeze(0)))
    embs = torch.cat(embs)
    names = np.array(names)
    torch.save(embs, conf.newfaces / 'cacheembedding.pth')
    np.save(conf.newfaces / 'names', names)


def update_cach_emd(conf, results_cach, new_embs):
    emds_cached = torch.load(conf.newfaces /'cacheembedding.pth')
    print("emds_cached shape:", emds_cached.shape)
    names = np.load(conf.newfaces / 'names.npy')
    new_id = int(names[-1][4:])
    for i, idx in enumerate(results_cach):
        if idx == -1:
            new_id += 1
            emds_cached = torch.cat((emds_cached, new_embs[i]), dim=0)
            names = np.append(names, 'new_' + str(new_id))
    # newemds = torch.cat((emds_cached, torch.tensor(new_emds, dtype=emds_cached.dtype)), dim=0)
    torch.save(emds_cached, conf.newfaces / 'cacheembedding.pth')
    np.save(conf.newfaces / 'names', names)


def load_cach(conf):
    embeddings = torch.load(conf.newfaces / 'cacheembedding.pth')
    names = np.load(conf.newfaces / 'names.npy')
    return embeddings, names


def infer_cache(conf, model, faces, target_embs, tta=False, threshold=1.57):
    embs = []
    for img in faces:
        if tta:
            mirror = trans.functional.hflip(img)
            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:
            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    source_embs = torch.cat(embs)
    diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)  # transpose--> (1, 512, n_boxes)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    minimum, min_idx = torch.min(dist, dim=1)
    min_idx[minimum > threshold] = -1  # if no match, set idx to -1
    return min_idx, minimum

    
def prepare_all():
    print("loading faceboxes model!!")
    faceboxes = FaceBoxesDetect(cfg)
    faceboxes.prepare()
    print('faceboxes loaded!!\n')
    
    print("prepare Arc model!!!")
    conf = get_config(False)
    conf.use_mobilfacenet = True
    
    learner = face_learner(conf, inference=True)
    learner.threshold = conf.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, '0.884.pth', from_save_folder=False, model_only=True)
    learner.model.eval()
    print('Arc loaded\n')

    if conf.update:
        targets, names = prepare_facebank(conf, learner.model, faceboxes, tta=conf.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    return conf, faceboxes, learner, targets, names


"""
##### test   infer_frame in facebox_arc_infer fucn #####
_, _, targets, names = prepare_all()  


def infer_frame(conf, faceboxes, learner, frame):
    dets = faceboxes.facebox_detect(frame)
    if dets.shape[0] == 0:
        print("no de")
        # bboxes, shoulders, faces = [], [], []
        return 
    else:
        bboxes, shoulders, faces = faceboxes.align_multi(frame, dets, save=True)
    if len(bboxes) == 0:
        print("filtered all faces which score < 0.7")
        return
    else:
        bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces  bboxes is list
        bboxes = bboxes.astype(int)
        # bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
        results, score = learner.infer(conf, faces, targets,
                                       True)  # results are indexs in the labface which detected face matched
        print("results:",results, "\n,score in one frame:", score)
        id_score = []    # predict id and match score
        new_faces = []
        new_faces_bbox = []
        new_embs = []
        for idx, bbox in enumerate(bboxes):
            # results[idx] : the idx boxes's index in the labface
            if results[idx] == -1:
                new_faces.append(faces[idx])
                new_faces_bbox.append(bbox)
                new_embs.append(learner.model(conf.test_transform(faces[idx]).to(conf.device).unsqueeze(0)))
                id_score.append([-1, 0])
            else:
                id_score.append([names[results[idx] + 1], score[idx]])
        if len(new_faces) != 0:
            if os.path.exists(conf.newfaces / 'cacheembedding.pth'):
                embed_cached, names_cached = load_cach(conf)
                results_cach, score_cach = infer_cache(conf, learner.model, new_faces, embed_cached, tta=False,
                                                       threshold=1.0)    # not best threshold value
                for idx, bbox in enumerate(new_faces_bbox):
                    for ids in id_score:
                        if ids[0] == -1:
                            id_score[0] = names_cached[results_cach[idx]]
                            id_score[1] = score_cach[idx]
                    
                if torch.any(torch.eq(results_cach, -1)):
                    update_cach_emd(conf, results_cach, new_embs)
            else:
                add_new_emd(conf, learner.model, new_faces)
        print(bboxes.shape, shoulders.shape)
        result = np.hstack((bboxes,shoulders[:, :-1]))
        print(result.shape, np.array(id_score).shape)
        result = np.hstack((result, np.array(id_score)))
        print(result.shape)
    return bboxes, shoulders[:, :-1], np.array(id_score)
"""


def facebox_arc_infer():
    conf, faceboxes, learner, targets, names = prepare_all()
    print("starting video capture")
    cap = cv2.VideoCapture(str(conf.video_path/conf.file_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps, type(fps))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter(str(conf.video_path / '_OoO_{}.avi'.format(conf.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), size)
    f_count = 0
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            """
            ##### test   infer_frame in facebox_arc_infer fucn #####
            x = infer_frame(conf, faceboxes, learner, frame)
            if x:
                bboxes_, shoulders_, id_score_ = x
                print("+++++===", id_score_, '\n', id_score_)
            else:
                print("no face")
            """
            f_count += 1
            # cv2.imshow('Video', frame)
            video_writer.write(frame)
            # image = Image.fromarray(frame[...,::-1])   # bgr to rgb
            # image = Image.fromarray(frame)
            dets = faceboxes.facebox_detect(frame)
            try:
                # bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)
                bboxes, _, faces = faceboxes.align_multi(frame, dets, save=True)
                """
                for inx, face in enumerate(faces):
                    cv2.imwrite('./data/{}{}.jpg'.format(np.random.randint(100), inx), np.array(face))
                print("bboxes", bboxes, type(bboxes), type(faces[0]))
                """
            except:
                bboxes = []
                faces = []
            if len(bboxes) == 0:
                # video_writer.write(frame)
                print('the {} frame detect no face'.format(f_count))
                continue
            else:
                # print("before infer", bboxes, faces!)
                bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces  bboxes is list
                bboxes = bboxes.astype(int)
                # bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
                results, score = learner.infer(conf, faces, targets,
                                               True)  # results are indexs in the labface which detected face matched
                print("results:",results, "\n,score in one frame:", score)
                new_faces = []
                new_faces_bbox = []
                new_embs = []
                for idx, bbox in enumerate(bboxes):
                    # results[idx] : the idx boxes's index in the labface
                    if results[idx] == -1:
                        new_faces.append(faces[idx])
                        new_faces_bbox.append(bbox)
                        new_embs.append(learner.model(conf.test_transform(faces[idx]).to(conf.device).unsqueeze(0)))
                        cv2.imwrite(os.path.join(conf.new_face, '{}.jpg'.format(
                            str(datetime.now())[:-7].replace(":", "_").replace(" ", "_"))), np.array(faces[idx]))
                    else:
                        if conf.score:
                            frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                        else:
                            frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                if len(new_faces) != 0:
                    if os.path.exists(conf.newfaces / 'cacheembedding.pth'):
                        embed_cached, names_cached = load_cach(conf)
                        results_cach, score_cach = infer_cache(conf, learner.model, new_faces, embed_cached, tta=False,
                                                               threshold=1.0)
                        for idx, bbox in enumerate(new_faces_bbox):
                            if conf.score:
                                frame = draw_box_name(bbox, names_cached[results_cach[idx]] + "_{:.2f}".format(score_cach[idx]), frame)
                            else:
                                frame = draw_box_name(bbox, names_cached[results_cach[idx]], frame)
                        if torch.any(torch.eq(results_cach, -1)):
                            update_cach_emd(conf, results_cach, new_embs)
                    else:
                        add_new_emd(conf, learner.model, new_faces)
                else:
                    # print("no new faces!!!")
                    pass
                # video_writer.write(frame)
            video_writer.write(frame)

        else:
            break
    
    cap.release()
    video_writer.release()


# camera = cv2.VideoCapture(' ')
# load model and others
# conf, faceboxes, learner, targets, names = prepare_all()

# infer_frame(frame)


def infer_video():
    pass
    

# facebox_arc_infer()



