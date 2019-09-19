import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2

import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import ONet


# img = torch.rand(1, 3, 48, 48)
# print(img.shape, img.max())
# net = ONet()
# out = net(img)
# print(out.shape)


def landmark_loss(gt_landmark, pred_landmark):
    loss_landmark = nn.MSELoss()
    pred_landmark = torch.squeeze(pred_landmark)
    gt_landmark = torch.squeeze(gt_landmark)
    # print(pred_landmark.shape, gt_landmark.shape, gt_landmark.dtype, pred_landmark.dtype)
    return loss_landmark(pred_landmark,gt_landmark)


class LandmarkData(data.Dataset):
    def __init__(self, face_landmark_48, transforms):
        super(LandmarkData, self).__init__()
        self.data_info = np.load(face_landmark_48)
        self.transforms = transforms

    def __getitem__(self, index):
        oneinfo = self.data_info[index]
        img = cv2.imread(oneinfo['filepath'])
        h, w, _ = img.shape
        landmark = oneinfo['points']
        landmark[0::2] /= w
        landmark[1::2] /= h
        landmark = np.array(landmark, dtype='float32')
        img = self.transforms(img)
        return img, landmark

    def __len__(self):
        return len(self.data_info)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset = LandmarkData('./data/face_five/face_five_crop_48_test.npy', transform)
# img, t = dataset.__getitem__(2)
# print(img.shape, t.shape)
dataloader = data.DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)
"""
for img, landmark in dataloader:
    print(img.shape, landmark.shape)
"""
# optimizer = optim.Adam()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
net = ONet()
#net = net.to(device)


def train_onet(model, dataloader, base_lr=1e-4, epoch=150):

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    for ep in range(epoch):
        for img, landmark in dataloader:
            img = img.to(device)
            landmark = landmark.to(device)
            optimizer.zero_grad()
            landmarks_pred = model(img)
            loss = landmark_loss(landmark, landmarks_pred)
            loss.backward()
            optimizer.step()
        print("epoch {}, loss is {}".format(ep, loss))
        if ep % 10 == 0 and ep > 0:
            torch.save(model.state_dict(), './weights/Onet_{}.pt'.format(ep))
            
#train_onet(net, dataloader)


def filter_face(p):
    k1 = (p[3] - p[1]) / (p[2] - p[0]) if (p[2] - p[0]) != 0 else 0
    k2 = (p[9] - p[7]) / (p[8] - p[6]) if (p[8] - p[2]) != 0 else 0
    k = (k1 + k2) / 2
    d1 = p[4] - (p[0]+p[6])/2
    d2 = (p[2]+p[8])/2 - p[4]
    scale = d1 / d2 if d2 != 0 else 100
    return k, scale

def infer(net, model_path):
    #print(net)
    if isinstance(net, nn.Module):
        pass
    net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device)))
    model =  net
    model.to(device)

    print(model)
    #model.eval()
    pred_points = []
    with torch.no_grad():
        """
        img = transform(img)
        scale = img.shape[-1]
        print(img.shape)
        img = img.unsqueeze(0).to(device)
        out = model(img)
        print(out.shape)
        points = out.squeeze().cpu().numpy()
        print(points, points.dtype)
        points *= scale
        points = list(map(np.ceil, points))
        points = list(map(int, points))
        print("points:", points)
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        print(img, img.shape)
        min_, max_ = img.min(), img.max()
        img = np.array(((img - min_) / (max_ - min_))*255, dtype=np.uint8)
        for i in range(0, 10, 2):
            img = cv2.circle(img, (points[i], points[i+1]), 3, (0, 255, 255), -1)
        cv2.imwrite('./img_pred.jpg', img)
        """
        for j, (img, landmark) in enumerate(dataloader):
            img = img.to(device)
            landmark = landmark.to(device)
            scale = img.shape[-1]
            out = model(img)
            img = img.unsqueeze(0).to(device)

            points = out.squeeze().cpu().numpy()
            points *= scale
            points = list(map(np.ceil, points))
            points = list(map(int, points))
            k, scale = filter_face(points)
            img = img.cpu().squeeze().permute(1, 2, 0).numpy()
            min_, max_ = img.min(), img.max()
            img = np.array(((img - min_) / (max_ - min_))*255, dtype=np.uint8)
            pred_points.append(points)
            for i in range(0, 10, 2):
                img = cv2.circle(img, (points[i], points[i+1]), 3, (0, 255, 255), -1)
                #cv2.imwrite('./pred_five_test/img_pred_{}.jpg'.format(j),img)
            if scale < 0.15 or scale > 7:
                print("k:", k)
                cv2.imwrite('./pred_five_test_filter/filter_{}.jpg'.format(j),
                        img)
        #np.save('./data/face_five/pred_points_test.npy', pred_points)

        

#img = cv2.imread('img_g.jpg')
#print(img.shape)
#infer(net, './weights/Onet_140.pt')


def exploare(pred_points):
    pp = np.load(pred_points)
    xs = pp[:, 0::2]
    ys = pp[:,1::2]
    scale = (xs[:,2] - (xs[:,0] + xs[:,3])/2)/((xs[:,1] + xs[:,4])/2 - xs[:,2])
    print(scale[:100])
    c_l = 0
    c_r = 0
    num = len(scale)
    for s in scale:
        if s < 0.1:
            c_l += 1
        if s > 9:
            c_r += 1
    r_l = c_l / num
    r_r = c_r / num
    r = (c_l + c_r) / num
    print("ration:{},r_l:{}, r_r:{}".format(r, r_l, r_r))
#exploare('./data/face_five/pred_points_test.npy')


class Points():
    def __init__(self):
        super(Points, self).__init__()
        pass
     
   
def load_onet(net, weights_path, device):
    net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(device)))
    net = net.to(device)
    return net


def points_infer(model, recrop_face, device):
    with torch.no_grad():
        img = transform(recrop_face)
        scale = img.shape[-1]
        img = img.unsqueeze(0).to(device)
        out = model(img)
        points = out.squeeze().cpu().numpy()
        print(points, points.dtype)
        points *= scale
        points = list(map(np.ceil, points))
        points = list(map(int, points))
        print("points:", points)
        """
        img = img.cpu().squeeze().permute(1, 2, 0).numpy()
        print(img, img.shape)
        min_, max_ = img.min(), img.max()
        img = np.array(((img - min_) / (max_ - min_))*255, dtype=np.uint8)
        for i in range(0, 10, 2):
            img = cv2.circle(img, (points[i], points[i+1]), 3, (0, 255, 255), -1)
        cv2.imwrite('./img_pred.jpg', img)
        """
    return points
    
