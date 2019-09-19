import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, log_sum_exp, match_
from data import cfg
GPU = cfg['gpu_train']


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + Î±Lloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by Î± which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        # num_classes, 0.35, True, 0, True, 7, 0.35, False
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))    # [21824, 4]  total number of anchor in one img if img.shape = (1024, 1024)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)    # LongTensor
        loc_pair = torch.Tensor(num, num_priors, 10)    # for 5 location points
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)    # key important
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)    # think more time
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # compute five key point loss  :
        # change the targets [4, 1, 10], predictions is three: loc_data, conf_data, loc_data_pair


        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0   # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + Î±Lloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
        
        
class MultiBoxLoss_(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        # num_classes, 0.35, True, 0, True, 7, 0.35, False
        super(MultiBoxLoss_, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, loc_five_data = predictions
        # print('loc_data shape:', loc_data.shape, '\n loc_five_data:', loc_five_data.shape)
        priors = priors
        num = loc_data.size(0)    #  batch size
        num_priors = (priors.size(0))    # [21824, 4]  total number of r anchor in one img if img.shape = (1024, 1024)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)    # LongTensor , class label for priors box match the gt
        loc_five = torch.Tensor(num, num_priors, 10)    # for 5 location points
        # print("before:", loc_t,conf_t,loc_five)
        for idx in range(num):
            # truths = targets[idx][:, :-1].data    # bbox, x1, y1, x2, y2: 0-1
            truths = targets[idx][:, :4].data  # bbox, x1, y1, x2, y2: 0-1
            truths_five = targets[idx][:, 5:].data    # coords 10
            labels = targets[idx][:, 4].data
            defaults = priors.data
            # match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)    # key important
            match_(self.threshold, truths, truths_five, defaults, self.variance, labels, loc_t, conf_t, loc_five, idx)  # key important
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            loc_five = loc_five.cuda()
        # loc_t is changed when math done!!!!
        # should loc_five change same time!!!!
        pos = conf_t > 0   # only optimizer positive anchors?
        print("pos conf_t:",conf_t.shape,  pos.shape, pos.sum())    # why all of pos is < 0????
        # print("after:",loc_t,conf_t, loc_five)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # print("pos_idx shape before:", pos.unsqueeze(pos.dim()).shape)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)    # think more time
        pos_idx_five = pos.unsqueeze(pos.dim()).expand_as(loc_five_data)
        # print("pos_idx shape after:", pos_idx.shape)
        loc_p = loc_data[pos_idx].view(-1, 4)    # choose positive loc_p from pred
        # print("loc_p:", loc_p)    # is empty....
        loc_t = loc_t[pos_idx].view(-1, 4)    # get correspond loc_t to loc_p which have matched
        loc_five = loc_five[pos_idx_five].view(-1, 10)
        loc_f = loc_five_data[pos_idx_five].view(-1, 10)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # print("loss_l:",loc_p, loc_t)
        loss_coords = F.mse_loss(loc_five, loc_f,  reduction='sum')
        # print("loss_coords:",loc_five, loc_f, loss_coords)

        # compute five key point loss  :
        # change the targets [4, 1, 10], predictions is three: loc_data, conf_data, loc_data_pair

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0   # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print("sum of neg:", neg.sum(), '\n', "sum of pos:", pos.sum())    # may zero !!!!!!!!
        
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # print("???????", conf_data.shape, '\n', conf_p, conf_p.shape)
        # conf_p may empty!!!!!!!!!!
        if conf_p.shape[0] == 0:
            print(pos_idx.shape, neg_idx.shape, conf_p)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        # print("?XX"*3)
        """
        try:
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        except:
            print("wwwwwww", targets_weighted.max())    # may is -9223372036854775808  ???
        
        """

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + ¦ÁLloc(x,l,g) + betaLloc_f(x,l,g_f)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_coords /= N

        return loss_l, loss_c, loss_coords
