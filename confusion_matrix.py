from data.data_pipe import get_train_dataset
from sklearn.metrics import confusion_matrix
#from resources.plotcm import plot_confusion_matrix

import torch
from apis.infer_on_video_cla import prepare_all
from torch.utils import data as data
from data.collate import detection_collate
from utils.utils_arc import prepare_facebank, load_facebank
import numpy as np

from scipy import stats


conf, faceboxes, learner, targets, names = prepare_all()

#print(conf.keys())


#ds, class_num = get_train_dataset(conf.emore_folder_val)
#print(ds, class_num)
#for i, (img, label) in enumerate(ds):
#    print(img.shape, label)
#    if i == 1:
#        break
"""
dataloader = data.DataLoader(ds, batch_size=100, shuffle=True,
        num_workers=4,collate_fn=detection_collate)

x = next(iter(dataloader))
print(x)
"""
"""
conf.update = True
conf.facebank_emds, conf.facebank_names = 'emds_train.pth', 'names_train.npy'
emds_val, names_val = prepare_facebank(conf, learner.model, faceboxes, tta=True)
#print(type(emds_val))
"""

"""
conf.facebank_path = conf.data_path/'faces_emore/val/'
conf.facebank_emds, conf.facebank_names = 'emds_val.pth', 'names_val.npy'
emds_val, names_val = prepare_facebank(conf, learner.model, faceboxes, tta=True)
"""

def bank_cm(emds_train,name_t, emds_val, name_v, th=1.52):
    #preds = []
    #name_t = name_t[1:]    # when name_t[0] is "Unknow"
    # val (m, 512)->(m, 512, 1), train(n, 512)->(1, 512, n)
    emds_train = emds_train.cpu().detach()
    emds_val = emds_val.cpu().detach()
    diff = emds_val.unsqueeze(-1) - emds_train.transpose(1, 0).unsqueeze(0)
    # diff (m, 512, n)
    
    dist = torch.sum(torch.pow(diff, 2), dim=1)    # (m, n)
    #minimum, min_idx = torch.min(dist, dim=1)    # (n_box), (n_box)
    minimums, min_idxs = torch.topk(dist, k=5, dim=1, largest=False)
    print("is [m, 5]?:",minimums.shape)
    min_idxs[minimums > th] = -1   # if no match, set idx to -1
    min_idxs = min_idxs.data.numpy()
    
    #counts = np.bincount(pred_labels)
    #np.argmax(counts)
    print('shape', type(name_t), name_t.shape)  # (100,)
    name_t = name_t.reshape(1, name_t.shape[0])

    print(name_t)
    name_texpand = np.tile(name_t, (name_t.shape[1], 1))
    #print(name_texpand)
    print(name_texpand.shape)
    print("name expand shape:", name_texpand.shape)
    print('min_idxs:', min_idxs.shape)
    #print(min_idxs, type(min_idxs[0][0]))
    print(min_idxs[0], name_texpand[0][min_idxs[0]], "[[[[[[[")
    print(min_idxs, min_idxs.shape,type(min_idxs[0][0]))

    pred_labels = [n_p[min_idx] for n_p, min_idx in zip(name_texpand, min_idxs)]

    print(pred_labels[0])
    #pred_labels = name_texpand[min_idxs[:,]]
    preds = [stats.mode(list(map(int, p)))[0][0] for p in pred_labels]
    #print(preds, '\n', name_v)
    name_v = np.array(name_v, dtype=np.int64)
    cm = confusion_matrix(name_v, preds)
    return cm
 
"""
emds_train, names_train = load_facebank(conf)

conf.facebank_path = conf.data_path/'faces_emore/val/'
conf.facebank_emds, conf.facebank_names = 'emds_val.pth', 'names_val.npy'
emds_val, names_val = load_facebank(conf)

cm = bank_cm(emds_train, names_train,emds_val, names_val)
sum_diag = 0
for i in range(cm.shape[0]):
    sum_diag += cm[i, i] 

print('right ratio is:', sum_diag / cm.shape[0])
print("confusion matrix is:", cm)
"""
