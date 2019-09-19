from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as datas
from data import detection_collate, preproc, preproc_, AFLW, VOCDetection, AnnotationTransform
from configs import cfg
from layers.modules import MultiBoxLoss, MultiBoxLoss_
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.faceboxes import FaceBoxes, FaceBoxesCoords
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='FaceBoxes  or FaceBoxesCoords Training')
parser.add_argument('--training_dataset', default='./data/WIDER_FACE', help='Training dataset directory')
parser.add_argument('--npy_file', default='./data/all_info_.npy', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=100, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights_detect/', help='Location to save checkpoint models')
parser.add_argument('--save_name',  help='what name do u want to save')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

img_dim = 1024   # only 1024 is supported ?
rgb_mean = (104, 117, 123)   # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
npy_file = args.npy_file
save_folder = args.save_folder
gpu_train = cfg['gpu_train']

net = FaceBoxes('train', img_dim, num_classes)
# net = FaceBoxesCoords('train', img_dim, num_classes)
print("Printing net...")
# print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]   # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
# criterion = MultiBoxLoss_(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def show_img_info(img_info):
    # when FaceBoxesCoords test
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
        cv2.rectangle(img, [np.array(polylines, np.int32), True], (0, 255, 255), 4)
    if not os.path.exits('./data/rect_coords'):
        os.mkdirs('./data/rect_coords')
    cv2.imwite(os.path.join('./data/rect_coords', info['filename']), img)
    # cv2.imshow()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())
    # dataset = AFLW(training_dataset, npy_file, preproc_(img_dim, rgb_mean))
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(datas.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), save_folder + '{}_'.format(args.save_name) + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        # print("trainning batch:", len(images), len(targets), targets[0].shape)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]
        

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        # loss_l, loss_c, loss_f = criterion(out, priors, targets)
        # loss = cfg['loc_weight'] * loss_l + loss_c + cfg['loc_five_weight'] * loss_f
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.4f} || Batchtime: {:.4f} s || ETA: {}'.format(epoch, max_epoch, (iteration % epoch_size) + 1, 
            epoch_size, iteration + 1, max_iter, loss_l.item(),loss_c.item(),lr, batch_time, str(datetime.timedelta(seconds=eta))))
        
    torch.save(net.state_dict(), save_folder + 'Final_{}_'.format(args.save_name))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

"""
if __name__ == '__main__':
    train()
"""
train()
