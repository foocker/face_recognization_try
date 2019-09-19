import numpy as np
import sqlite3
import pandas as pd
import time
import cv2
import os

def get_sqlitinfo(sqpath):
    # sqpath = "E:\\BaiduNetdiskDownload\\AFLW\\aflw\\data\\aflw.sqlite"
    conn = sqlite3.connect(sqpath)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    Tables = cur.fetchall()
    for table in Tables:
        print(table)
        table_name = table[0]
        print("table_name!!!", table_name)
        cur.execute("SELECT * FROM {}".format(table_name))
        col_name_list = [tuple[0] for tuple in cur.description]
        print("col_name_list!!", col_name_list)
    conn.close()


def face_rec_coord_sqlit(sqpath):
    print('start sparse!!')
    with sqlite3.connect(sqpath) as con:
        df_Faces = pd.read_sql_query("SELECT face_id,file_id FROM Faces", con)
        df_FaceImages = pd.read_sql_query(
            "SELECT image_id, db_id, file_id, filepath, bw, width, height FROM FaceImages", con)
        print(df_FaceImages)
        df_FaceRect = pd.read_sql_query("SELECT face_id,x,y,w,h,annot_type_id FROM FaceRect", con)
        df_FeatureCoords = pd.read_sql_query("SELECT face_id,feature_id,x,y,annot_type_id FROM FeatureCoords", con)

    result = set(df_FaceRect['face_id']).intersection(set(df_FeatureCoords['face_id']))
    split_Rect = df_FaceRect.iloc[:, :5]
    split_FeatureCoords = df_FeatureCoords.iloc[:, :4]
    all_info = []
    for fd in result:
        name = df_Faces[df_Faces.face_id == fd]['file_id'].values[0]
        filepath =  df_FaceImages[df_FaceImages.file_id==name]['filepath'].values[0]
        bbox = split_Rect[split_Rect.face_id == fd].values[0][1:]
        if np.min(bbox) < 0:
            # print(bbox)
            continue
        bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]    # change to x1, y1, x2, y2
        Coords = split_FeatureCoords[split_FeatureCoords.face_id == fd]
        ch_7 = Coords.loc[Coords['feature_id'] == 7]
        ch_12 = Coords.loc[Coords['feature_id'] == 12]
        ch_15 = Coords.loc[Coords['feature_id'] == 15]
        ch_18 = Coords.loc[Coords['feature_id'] == 18]
        ch_20 = Coords.loc[Coords['feature_id'] == 20]
        ch_C = pd.concat((ch_7, ch_12, ch_15, ch_18, ch_20), axis=0)
        if ch_C.shape[0] != 5:
            continue
        ch_CC = (ch_C.iloc[:, 2:].values).flatten()
        # print(ch_C.iloc[:, 2:].values)
        # print(Coords, "\n", ch_C, ch_C.shape[0],'\n', ch_CC)
        if filepath.endswith('.png'):
            filepath = filepath.replace('png', 'jpg')
        all_info.append({'filename':filepath, 'bbox': bbox, 'coords': ch_CC})

    # with open('all_info.json', 'w+') as f:
    #     # TypeError: Object of type 'int64' is not JSON serializable
    #     json.dump(all_info, f)
    np.save('all_info_.npy', all_info)


def show_img_info(img_info):
    all_info = np.load(img_info)
    c = 0
    for info in  all_info:
        x1, y1, x2, y2 = list(map(int, info['bbox']))
        # [px1, py1, px2, py2, px3, py3, px4, py4, px5, py5] = list(map(int, info['coords']))
        coords = list(map(int, info['coords']))
        img = cv2.imread(os.path.join('./data/flickr', info['filename']))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        polylines = []
        for i in range(0, 10, 2):
            polylines.append([coords[i], coords[i+1]])
            cv2.circle(img, (coords[i], coords[i+1]), 5, (0, 255, 0), -1)
        cv2.polylines(img, [np.array(polylines, np.int32)], True, (0, 255, 255), 4)
    if not os.path.exits('./data/rect_coords'):
        os.mkdirs('./data/rect_coords')
    cv2.imwrite(os.path.join('./data/rect_coords', info['filename']), img)
    # cv2.imshow()




sqpath = "./alfw_rect_coords/aflw.sqlite"
st = time.time()
face_rec_coord_sqlit(sqpath)
print("cost time on face_rec_coord_sqlit is: ", time.time()-st)


# show_img_info('all_info_.npy')
