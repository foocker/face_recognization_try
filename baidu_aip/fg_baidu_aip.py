# coding=utf-8
import base64
import cv2
import numpy as np
from aip import AipFace
import time
import math
import os

API_KEY = ''    # your own key

SECRET_KEY = ''    # your own s-key


def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()


def npbase64(frame):
    retval, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode()
    return img_str


def base64np(bas_str):
    img_d = base64.b64decode(bas_str)
    nparr = np.fromstring(img_d, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def decorate(fun):
    count = 0

    def wrapper(*args, **kwargs):
        nonlocal count
        start_time = time.time()
        data = fun(*args, **kwargs)
        stop_time = time.time()
        dt = stop_time - start_time
        count += 1
        print(fun.__name__, "call %d times，cost time %f。" % (count, dt))
        return data
    return wrapper


class FaceRecognization(AipFace):
    def __init__(self, match_threshold, max_user_num, max_face_num, base_group, QF=True):
        self.QF = QF
        self.APP_ID = ''   # your own app_id
        self.imageType = "BASE64"
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.base_group = base_group
        self.max_user_num = max_user_num
        self.max_face_num = max_face_num    # 1:3 .0; 2: 4.5; 3: 6.8
        self.match_threshold = match_threshold
        self.client = AipFace(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        # self.base_userids = self.client.getGroupUsers(self.base_group)['result']['user_id_list']

    def encode_img(self, info):
        if isinstance(info, str):
            img_bin = read_file(info)
            try:
                str_img = base64.b64encode(img_bin).decode()
                return str_img
            except:
                print("b64encode img fail")
                return None
        elif isinstance(info, np.ndarray):
            try:
                img_str = npbase64(info)
                return img_str
            except:
                print("encode ndarray fail")
                return None
        else:
            print("input info not support")
            return None

    def check_location(self, l, t):
        l = l if l > 0 else 0
        t = t if t > 0 else 0
        return l, t

    def draw_ploylines(self, img, coords_):
        # coords = list(map(int, coords))
        for coords in coords_:
            polylines = []
            for i in range(0, 8, 2):
                polylines.append([coords[i], coords[i + 1]])
                img = cv2.circle(img, (coords[i], coords[i + 1]), 5, (0, 255, 0), -1)
                # img = cv2.polylines(img, [np.array(polylines, np.int32)], True, (0, 255, 255), 4)
        return img

    def draw_box(self, img, bboxes):
        # bbox = list(map(int, bbox))
        for bbox in bboxes:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2]+bbox[0], bbox[3]+bbox[1]), (0, 0, 255), 2)
        return img

    def quality_coords(self, bbox, coords, th_cos=15):
        # th_cos small will converge -1 :decrease
        x1, y1, w, h = bbox
        x2, y2 = x1+w, y1+h
        xs = coords[0::2]
        ys = coords[1::2]
        coordx_min , coordy_min= min(xs), min(ys)
        coordx_max, coordy_max = max(xs), max(ys)
        flagl = coordx_min > x1 and coordy_min > y1
        flagr = coordx_max < x2 and coordy_max < y2
        flaglocat = flagl and flagr
        vec31 = (coords[0] - coords[4], coords[1] - coords[5])
        vec34 = (coords[4] - coords[6], coords[5] - coords[7])
        vec32 = (coords[2] - coords[4], coords[3] - coords[5])
        cos134 = (vec31[0]*vec34[0] + vec31[1]*vec34[1]) / math.sqrt((vec31[0]**2 + vec31[1]**2) * (vec34[0]**2+vec34[1]**2))
        cos234 = (vec32[0]*vec34[0] + vec32[1]*vec34[1]) / math.sqrt((vec32[0]**2 + vec32[1]**2) * (vec34[0]**2+vec34[1]**2))
        flagcos = math.cos(3.1415/180*th_cos) > max(abs(cos134), abs(cos234))
        print("+++++++", flaglocat, cos134, cos234, flagcos)

        return flaglocat and flagcos

    @decorate
    def quality_filter(self, detect_result, oc_thd=0.6, fb_thd=0.8, eye_thd=0.4, ck_thd=0.55, chin_thd=0.55, bu_thd=0.6,
                       TD=20, rot=15, illum_thd=70, area_range=[40, 200]):
        """
        :param detect_result: from detect_function
        :param oc_thd: 0.6, 0.7, 0.8 diff part should it diff range, but here is one
        :param bu_thd: < 0.7
        :param illum_thd: > 40
        :param area_range: 80-200
        :return: filtered boxes
        """
        filtered_boxes = []
        filtered_info = []
        filtered_coords = []
        face_list = detect_result['result']['face_list']
        for finfo in face_list:
            fb = finfo['face_probability'] > fb_thd
            if not fb:
                continue
            landmarks = finfo['landmark']
            x1, y1 = round(landmarks[0]['x']), round(landmarks[0]['y'])
            x2, y2 = round(landmarks[1]['x']), round(landmarks[1]['y'])
            x3, y3 = round(landmarks[2]['x']), round(landmarks[2]['y'])
            x4, y4 = round(landmarks[3]['x']), round(landmarks[3]['y'])
            coords = [x1, y1, x2, y2, x3, y3, x4, y4]
            h, w = finfo['location']['height'], finfo['location']['width']
            rotation = abs(finfo['location']['rotation'])
            l, t = int(finfo['location']['left']), int(finfo['location']['top'])
            # some pred loct is negative #
            l, t = self.check_location(l, t)
            finfo['location']['left'], finfo['location']['top'] = l, t
            # some pred loct is negative #
            occlusion = finfo['quality']['occlusion']
            blur = finfo['quality']['blur']
            illumination = finfo['quality']['illumination']
            completeness = finfo['quality']['completeness']
            ThreeD = max(list(map(abs, list(finfo['angle'].values())))) < TD
            # flag_oc = np.all(np.array(list(occlusion.values())) < oc_thd)
            check_flag = max(occlusion['left_cheek'], occlusion['right_cheek']) < ck_thd
            mouth_chin_flag = max(occlusion['mouth'], occlusion['chin_contour']) < chin_thd
            eye_flag = max(occlusion['left_eye'], occlusion['right_eye']) < eye_thd
            flag_oc = check_flag and mouth_chin_flag and eye_flag

            flag_bu = blur < bu_thd
            flag_illum = illumination > illum_thd
            flag_area = (area_range[0] < h < area_range[1]) and (area_range[0] < w < area_range[1])
            flag_rota = rotation <= rot
            flagcoord = self.quality_coords([l, t, w, h], coords)
            # flag_all = np.all([flag_oc, flag_bu, flag_illum, flag_area])
            if flag_oc and flag_bu and flag_illum and flag_area and\
                    flag_rota and ThreeD and completeness and flagcoord:
                filtered_boxes.append([l, t, w, h])
                filtered_coords.append(coords)
                filtered_info.append(finfo)
        return filtered_boxes, filtered_info, filtered_coords

    def shoulder_face(self, img, box, ratio=1.5, h_r=0.7):
        h, w, _ = img.shape
        x1, y1, wb, hb = box
        scale = max(wb, hb) / min(wb, hb)
        expand = scale * ratio
        wb_e, hb_e = wb * expand, hb * expand
        x1_ = x1 - (wb_e - wb)/2 if x1 - (wb_e - wb)/2 > 0 else 1
        y1_ = y1 - h_r*(hb_e - hb) if y1 - h_r*(hb_e - hb) > 0 else 1
        x2_ = x1 + wb + (wb_e - wb)/2 if x1 + wb + (wb_e - wb)/2 < w else w-1
        y2_ = y1 + hb + (1-h_r)*(hb_e - hb) if y1 + hb + (1-h_r)*(hb_e - hb) < h else h-1
        return [round(x1_), round(y1_), round(x2_ - x1_), round(y2_-y1_)]

    def noraml_face(self, img, bbox, size=240):
        # img is crop from frame by bbox
        # normal cropped img's shape to 240
        x1, y1, w, h = bbox
        # h_0, w_0, _ = img.shape
        max_ = max(h, w)
        ratio = size / max_
        n_h, n_w = int(ratio * h + 0.5), int(ratio * w + 0.5)
        assert n_h > 0 and n_w > 0
        assert n_h == size or n_w == size
        img_resized = cv2.resize(img, (n_w, n_h))
        img_container = np.zeros((size, size, 3), dtype=np.uint8)
        if n_h == size:
            left = size - n_w
            print(img_resized[:, :, 0].shape, left)
            img_container[:, :, 0] = np.pad(img_resized[:, :, 0], ((0, 0), (left, 0)), 'constant',
                                            constant_values=(0, 0))
            img_container[:, :, 1] = np.pad(img_resized[:, :, 1], ((0, 0), (left, 0)), 'constant',
                                            constant_values=(0, 0))
            img_container[:, :, 2] = np.pad(img_resized[:, :, 2], ((0, 0), (left, 0)), 'constant',
                                            constant_values=(0, 0))
        else:
            top = size - n_h
            print(img_resized[:, :, 0].shape, top)
            img_container[:, :, 0] = np.pad(img_resized[:, :, 0], ((top, 0), (0, 0)), 'constant',
                                            constant_values=(0, 0))
            img_container[:, :, 1] = np.pad(img_resized[:, :, 1], ((top, 0), (0, 0)), 'constant',
                                            constant_values=(0, 0))
            img_container[:, :, 2] = np.pad(img_resized[:, :, 2], ((top, 0), (0, 0)), 'constant',
                                            constant_values=(0, 0))
        return img_container


    @decorate
    def detect_client(self, img, options={}):
        options["max_face_num"] = self.max_face_num
        options['face_field'] = 'quality,age,gender,emotion,glasses,landmark'    # default settings

        encoded_str = self.encode_img(img)
        if encoded_str is not None:
            result = self.client.detect(encoded_str, self.imageType, options)
            try:
                if result['result'] is not None:
                    return result
            except:
                print("detect result:", result, "type of input", type(img))
                # raise ValueError('result is wrong? for input image is not right')
                return
        else:
            return encoded_str

    def check_cache_group(self):
        #  '_cache' group
        grouplist =  self.client.getGroupList()['result']['group_id_list']
        if len(grouplist) == 1:
            return False
        else:
            for gp in grouplist:
                if 'cache' in gp:
                    return True
            return False

    def add_group(self, name='cache'):
        try:
            self.client.groupAdd(name)
        except:
            print('this group exist!')

    def crop_box(self,frame, locs):
        boxes = [frame[int(loc[1]):int(loc[1]+loc[3]),int(loc[0]):int(loc[0]+loc[2])] for loc in locs]
        return boxes

    def sparse_search_result(self, result, filtered_info, boxes):
        # input must have valid information
        result_ = {}
        need_result = []
        if result['result'] is not None:
            for i, info in enumerate(filtered_info):
                temp = {}
                face_token = filtered_info[i]['face_token']
                # print("face_token",face_token)
                face_list = result['result']['face_list']
                for info_ in face_list:
                    if info_['face_token'] == face_token:
                        if len(info_['user_list']) != 0:
                            temp['group_id'] = info_['user_list'][0]['group_id']  # choose the first matched
                            temp['user_id'] = info_['user_list'][0]['user_id']  # may []
                            if len(info_['user_list']) > 1:
                                tt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                with open('facerecognition/user_list.txt', 'a+') as f:
                                    f.write(tt+str(info_['user_list'])+'\n')
                        else:
                            temp['group_id'], temp['user_id'] = None, None

                        temp['location'] = list(map(int, info['location'].values()))[:-1]    # info is filter_detect
                        l, t = temp['location'][:2]
                        l, t = self.check_location(l, t)
                        temp['location'][:2] = l, t

                        temp['age'] = info['age']
                        temp['gender'] = info['gender']['type']
                        temp['emotion'] = info['emotion']['type']
                        temp['galsses'] = info['glasses']['type']
                        temp['face_normal'] = boxes[i]
                        need_result.append(temp)
                        break
        else:
            need_result = boxes

        result_['need_result'] = need_result
        return result_

    @decorate
    def search_result(self, frame, gp_id_list=["x1,x2", "x3", "x4"], op={}):
        """
        :param frame:
        :param gp_id_list: ["x1,x2", "x1", "x2"], should contain all group that add new face
        :param op: options for Search
        :return:
        """
        frame_copy = frame.copy()
        std = time.time()
        detect = self.detect_client(frame)
        print("detect time is ", time.time()-std)
        print("detect", detect)
        if detect is not None:
            filter_detect, filtered_info, filtered_coords = self.quality_filter(detect)
            print("filter:{}".format(len(filtered_info)), filtered_info)
            if len(filtered_info) != 0:
                shoulder_boxes = [self.shoulder_face(frame, fd) for fd in filter_detect]
                # boxes = self.crop_box(frame, filter_detect)
                boxes = self.crop_box(frame, shoulder_boxes)
                boxes_noraml = [self.noraml_face(bs, sb) for (bs, sb) in zip(boxes, shoulder_boxes)]
                # boxes = self.crop_box(frame, filter_detect)
                op = {"match_threshold": self.match_threshold, "max_user_num": self.max_user_num,
                      "max_face_num": self.max_face_num}
                st = time.time()
                result = self.client.multiSearch(self.encode_img(frame), self.imageType,
                                                 group_id_list=gp_id_list[0], options=op)
                tt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(tt, result)
                with open('facerecognition/detect_result.txt', 'a+') as f:
                    f.write(tt + str(result['result']) + '\n')
                print("multiSearch cost time", time.time()-st)
                print('result', result)
                result_ = self.sparse_search_result(result, filtered_info, boxes_noraml)
                print(result_)
                bbox = result_['need_result']
                if not isinstance(bbox[0], np.ndarray):
                    img = self.draw_box(frame_copy, bbox[0]['location'])
                    cv2.imwrite('facerecognition/img_box_{}.jpg'.format(tt), img)
                    print(img.shape)
                else:
                    for i, img_ in enumerate(bbox):
                        cv2.imwrite('facerecognition/face_update_{}_{}.jpg'.format(tt, i), img_)
                        print(img_.shape)
                return result_
            else:
                print('no good face')
                return
        else:
            print('no face detected')
            return

    def check_user_id_list(self, group_id):
        cacheusers = self.client.getGroupUsers(group_id)['result']['user_id_list']
        if len(cacheusers):
            max_id = int(cacheusers[-1][2:])  # sorted
        else:
            max_id = 0
        return max_id

    def face_count(self,user_id, group_id):
        try:
            ft = len(self.client.faceGetlist(user_id, group_id)['result']['face_list'])
        except (Exception, KeyError, RuntimeError) as e:
            print("face_count:", e)
            ft = -1

        return ft

    def draw_box(self, img, bbox):
        # bbox = list(map(int, bbox))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2]+bbox[0], bbox[3]+bbox[1]), (0, 0, 255), 2)
        return img

    def star_img(self,f1, f2):
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        h1, w1= img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img = np.zeros((2*h1+h2, 2*w1+w2, 3), dtype=np.uint8)
        img[:h1,:w1,:] = img1
        img[h1:h1+h2, w1:w1+w2, :] = img2
        img[:h1,w1+w2:,:] = img1
        img[h1+h2:, :w1, :] = img1
        img[h1+h2:,w1+w2:,:] = img1
        cv2.imwrite('tlp_fan.jpg', img)

    @decorate
    def update_cache_base(self, frame, matched_result, face_thr=6):
        if matched_result is None:
            print("no good, detected face")
            return
        if isinstance(matched_result['need_result'][0], np.ndarray):
            max_id = self.check_user_id_list('cache')
            print("max_user_id_list in cache", max_id)
            for i, box_img in enumerate(matched_result['need_result']):
                self.client.addUser(self.encode_img(box_img), self.imageType, 'cache',
                                    'c_' + str(max_id + i+1))
                print('add new user id  to user_id :{} to cache'.format('c_' + str(max_id + i+1)))
            return

        for info in matched_result['need_result']:
            box_img = info['face_normal']
            group_id, user_id = info['group_id'], info['user_id']  # default first
            if group_id and user_id is None:
                max_id = self.check_user_id_list('cache')
                print('add new user id  to cache', max_id + 1)
                self.client.addUser(self.encode_img(box_img), self.imageType, 'cache',
                                    'c_' + str(max_id + 1))
            else:
                f_t = self.face_count(user_id, group_id)
                if f_t == -1:
                    f_t = self.face_count(user_id, group_id)
                print("f_t", f_t)
                if f_t <= face_thr:
                    print("update in exist lab face num is {}, user_id:{},group_id:{}".format(f_t, user_id, group_id))
                    if group_id == 'cache':
                        op = {"user_info": 'has identified in cache group'}  # can add a count number
                        self.client.addUser(self.encode_img(box_img), self.imageType, group_id, user_id, options=op)
                        print("Updated identified in cache")
                    else:
                        self.client.addUser(self.encode_img(box_img), self.imageType, group_id, user_id)
                        print("Updated identified group_id:{}, user_id{}".format(group_id, user_id))
        return matched_result

    def test_video_prime(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps:", fps, type(fps))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        save_name = 'test'
        video_writer = cv2.VideoWriter('{}_OoO.avi'.format(save_name),
                                       cv2.VideoWriter_fourcc(*'XVID'), int(fps), size)
        f_count = 0
        wait_time = int(1.5*fps)
        while cap.isOpened():
            isSuccess, frame = cap.read()
            if isSuccess:
                if f_count % wait_time == 0:
                    # video_writer.write(frame)
                    cv2.imshow('Frame', frame)
                    matched_result = self.search_result(frame, gp_id_list=["fq0,cache","fq0"])
                    if matched_result is not None:
                        # or remove if

                        self.update_cache_base(frame, matched_result)
                        tt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        with open('result0.txt', 'a+') as f:
                            f.write(tt + str(matched_result['need_result']) + '\n')
                    else:
                        print("no good face or no face !")
                    # print("matched results:", matched_result)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                f_count += 1
            # Break the loop
            else:
                break
        # When everything done, release the video capture object
        video_writer.write(frame)
        cap.release()
        video_writer.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def test_video(self, frame):
        # frame_copy = frame.copy()
        matched_result = self.search_result(frame, gp_id_list=["x1,x2", "x1"])
        if matched_result is not None:
            tt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(tt, type(tt), str(matched_result['need_result']))
            with open('facerecognition/result.txt', 'a+') as f:
                f.write(tt + str(matched_result['need_result'])+'\n')
            print("need_result", matched_result['need_result'])
            return self.update_cache_base(frame, matched_result)

        else:
            print("no good face or no face or no matched face!")
            return None

    def diff_person(self, face_dir, group_id):
        # return the repeatedly person in images
        score_th = 70
        results = []
        for i, name in enumerate(os.listdir(face_dir)):  # if all image is diff person, otherwise, dir, and user_id
            img = cv2.imread(os.path.join(face_dir, name))
            try:
                img_str = self.encode_img(img)
            except:
                print("error")
                continue
            op = {"match_threshold": self.match_threshold, "max_user_num": self.max_user_num,
                  "max_face_num": self.max_face_num}
            # print(op)
            result_temp = self.client.search(img_str, self.imageType, group_id_list=group_id, options=op)

            print(result_temp)
            try:
                result = result_temp['result']
                print(result)
                if result and result['user_list'][0]['score'] >= score_th:
                    usr_id = result['user_list'][0]['user_id']
                    results.append({name: usr_id})
                    print(results)
            except:
                print("except :", result)
                continue

            print("add{}".format(name))
            self.client.addUser(self.encode_img(os.path.join(face_dir, name)), self.imageType, group_id=group_id,
                                user_id=name[:-4])
            time.sleep(0.5)    # 0.2
            print(results)
        np.save("twotwo_threshold_70.npy", results)









