from data.data_pipe import get_train_dataset
import torch.utils.data as data
import torch
import numpy as np
from models.arcface import MobileFaceNet
from scipy import stats
from sklearn.metrics import confusion_matrix

device = 'cuda:0'

ds_t, _ = get_train_dataset('./data/FBA_Friends_Folder/train')
ds_v, _ = get_train_dataset('./data/FBA_Friends_Folder/val')

s = ds_t.__len__()
dataloader = data.DataLoader(ds_t, batch_size=1)


model = MobileFaceNet(512)
model.load_state_dict(torch.load('./weights/model_mobile_0.96.pth'))
model.to(device).eval()


def get_emds_from_folder(model): 
        emds = torch.zeros([s, 512]).cuda()
        print(emds.shape)
        names = []
        for i, (img, label) in enumerate(dataloader):
            #print(img.shape)
            with torch.no_grad():
                emd = model(img.to(device))
                #emds.append(emd)
                emds[i] = emd
                names.append(label)
        #emds = np.array(emds)
        torch.save(emds, './data/FBA_Friends_Folder/emds_train.pth')
        #np.save('./data/FBA_Friends_Folder/val_emds.npy', emds)
        np.save('./data/FBA_Friends_Folder/names_train.npy', names)


#get_emds_from_folder(model)
e_t = torch.load('./data/FBA_Friends_Folder/train_emds.pth')
e_v = torch.load('./data/FBA_Friends_Folder/val_emds.pth')
n_t = np.load('./data/FBA_Friends_Folder/train_names.npy')
n_v = np.load('./data/FBA_Friends_Folder/val_names.npy')

def bank_cm(emds_train,name_t, emds_val, name_v, th=1.52):
    #preds = []
    #name_t = name_t[1:]    # when name_t[0] is "Unknow"
    # val (m, 512)->(m, 512, 1), train(n, 512)->(1, 512, n)
    emds_train = torch.Tensor(emds_train)
    emds_val = torch.Tensor(emds_val)
    diff = emds_val.unsqueeze(-1) - emds_train.transpose(1, 0).unsqueeze(0)
    print('diff shape', diff.shape)
    # diff (m, 512, n)
    
    dist = torch.sum(torch.pow(diff, 2), dim=1)    # (m, n)
    print("dist shape:", dist.shape)
    minimums, min_idxs = torch.topk(dist, k=3, dim=1, largest=False)
    #print("is [m, 5]?:",minimums.shape)
    min_idxs[minimums > th] = -1   # if no match, set idx to -1
    min_idxs = min_idxs.data.numpy()
    
    #counts = np.bincount(pred_labels)
    #np.argmax(counts)
    #print('shape', type(name_t), name_t.shape)  # (100,)
    name_t = name_t.reshape(1, name_t.shape[0])

    print("name_t", name_t.shape)
    name_texpand = np.tile(name_t, (name_t.shape[1], 1))
    #print(name_texpand)
    print("expand shape",name_texpand.shape)
    """
    print("name expand shape:", name_texpand.shape)
    print('min_idxs:', min_idxs.shape)
    #print(min_idxs, type(min_idxs[0][0]))
    print(min_idxs[0], name_texpand[0][min_idxs[0]], "[[[[[[[")
    print(min_idxs, min_idxs.shape,type(min_idxs[0][0]))
    """
    pred_labels = [n_p[min_idx] for n_p, min_idx in zip(name_texpand, min_idxs)]

    print("pred labels:",len(pred_labels))

    #pred_labels = name_texpand[min_idxs[:,]]
    preds = [stats.mode(list(map(int, p)))[0][0] for p in pred_labels]
    print(len(preds))
    #print(preds, '\n', name_v)
    name_v = np.array(name_v, dtype=np.int64)
    print(len(name_v),len(preds) )
    cm = confusion_matrix(name_v, preds)
    return cm  

cm = bank_cm(e_t, n_t, e_v, n_v)
print(cm)

