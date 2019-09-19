from torch.utils.data import  ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import  ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np


def de_preprocess(tensor):
    return tensor*0.5 + 0.5


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder)
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 


def get_val_data(data_path, stop_num=5000):
    # stop_num may out of error
    ds = ImageFolder(data_path, transform=None)
    imgs = []
    labels = []
    for i, v in enumerate(ds):
        # v[0] is Pil Image
        imgs.append(np.transpose(np.asarray(v[0]), (1, 2, 0)))
        labels.append(v[1])    # v[1] + 1 ?
        if i == stop_num:
            break
    return np.array(imgs), np.array(labels)

