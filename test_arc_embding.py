from data.prepare_valdata import get_val_dataset
from apis.learner import face_learner
from configs.config_arc import get_config
from utils.utils_arc import load_facebank
import os
import torch
from data.prepare_valdata import show_pair



device = 'cuda:0'

def prepare_arc(): 
    conf = get_config(False)
    conf.use_mobilfacenet = True

    learner = face_learner(conf, inference=True)
    learner.threshold = conf.threshold
    learner.load_state(conf, conf.arc_model, from_save_folder=False, model_only=True)
    
    learner.model.eval()
    print('Arc loaded\n')

    print('begin load facebank!')
    targets, names = load_facebank(conf)
    print('facebank loaded')
    return conf, learner, targets, names

def test_embding():
    conf, learner, targets, names = prepare_arc()
    learner.model.to(device).eval()
    print(targets.shape, names, len(names))
    val_data, val_label = get_val_dataset(conf.emore_folder_val)
    val_data = torch.Tensor(val_data).to(device)
    val_label = torch.Tensor(val_label).to(device)
    print('val_datashape', val_data.shape, val_label.shape)
    val_data_o = val_data[1::2]
    val_data_e = val_data[0::2]
    print(val_label, sum(val_label))

    embs_same, embs_diff = [], []
    i = 0
    for o, e, f in zip(val_data_o, val_data_e, val_label):
        i += 1
        #print(o.shape)
        emd1, emd2 = learner.model(o.unsqueeze(0)), learner.model(e.unsqueeze(0))
        diff = torch.pow(emd1-emd2, 2).sum().cpu().detach().numpy()
        #print(diff)
        if f:
            embs_same.append(diff)
            if diff > 1.52:
                show_pair(val_data.cpu().detach().numpy(),
                        val_label.cpu().detach().numpy(), i-1)
        else:
            embs_diff.append(diff)

        if i == 300:
            break
    print(embs_same[:30], '\n', embs_diff[:30],'\n','max is ',
            max(embs_same),'min is: ', min(embs_diff),
            sum(embs_same)/len(embs_same), sum(embs_diff)/len(embs_diff))
    
    #accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, val_data, val_label)

    #return accuracy, best_threshold, roc_curve_tensor


#accuracy, best_threshold, roc_curve_tensor = test_embding()
#print('acc:', accuracy, best_threshold, roc_curve_tensor)  # 0.96, 1.522

test_embding()
#img_A = 

