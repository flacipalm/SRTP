import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
Tensor=torch.tensor
from typing import Optional, Tuple, Any, List
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from utils import *
import utils
from dataset import KeyDataset
from model import *
from loss import LossFunc
import time
from configs import parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:1,2,3')
if __name__ == "__main__":
    
    t = time.localtime()
    start_time = time.time()
    args = parse_args()
    print(args)

    if t.tm_hour >= 13:         # change from Beijing's time to Chicago's time
        hour_mod = t.tm_hour - 13
        day_mod = t.tm_mday
    else:
        hour_mod = t.tm_hour + 10
        day_mod = t.tm_mday - 1
    month_mod = str(t.tm_mon)
    day_mod = str(day_mod)
    hour_mod = str(hour_mod)
    min_mod = str(t.tm_min)
    sec_mod = str(t.tm_sec)
    if len(month_mod) == 1:
        month_mod = "0" + month_mod
    if len(day_mod) == 1:
        day_mod = "0" + day_mod
    if len(hour_mod) == 1:
        hour_mod = "0" + hour_mod
    if len(min_mod) == 1:
        min_mod = "0" + min_mod
    if len(sec_mod) == 1:
        sec_mod = "0" + sec_mod

    #logger_file = open('./new_logger/{}{}{}_{}:{}:{}.txt'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec), 'w')
    logger_file = open('./logger{}/{}{}{}_{}:{}:{}_scene:{}_view:{}_lr:{}_eva:{}_epoch:{}_Bbox:{}_Hungarian:{}_Loss:{}.txt'.format(args.save, t.tm_year, month_mod, day_mod, hour_mod, min_mod, sec_mod, args.scene, args.view, args.lr, args.eva, args.epoch, args.bbox, args.hungarian, args.loss_comb), 'w')
    dataset = KeyDataset(scene=args.scene, view=args.view, Bbox=args.bbox, Hungarian=args.hungarian)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=args.shuffle, num_workers=0)
    #pdb.set_trace()
    
    model = TrackingMatch(Bbox = args.bbox).to(device).double()
    criterion = LossFunc(Bbox=args.bbox, Loss_comb=args.loss_comb)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
    epochs = args.epoch
    #pdb.set_trace()
    epoch_avg_loss = []
    epoch_avg_fploss = []   # foot_point_loss
    epoch_avg_tkloss = []   # tracking_loss
    if args.eva < 1:
        eva_frame = args.eva * len(train_loader)
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_total_fploss = 0
        epoch_total_tkloss = 0
        enum_times = 0
        for idx, item in enumerate(train_loader):
            # forward propagation
            # shape of input: (1, num_pairs, 17, 6) or (1, num_pairs, 3, 6)---when input bbox
            # shape of ground_truth: (num_pairs)
            input_orig, ground_truth, _ = item  # don't need pair_shape
            if args.bbox == True:
                input = input_orig[:,:,0:2,:]       #(1,num_pairs,2,6)
                bbox_all_foot_point = input_orig[:,:,2,:].squeeze(0)       #(1,num_pairs, 6) -> (pairs, 6)
                tracking_pred, homo_mat = model(input)       # transform from float to double
                tracking_pred = tracking_pred.squeeze(0)                #(1, num_pairs, 2) -> (num_pairs, 2)
                homo_mat = modify_homo_mat(homo_mat)
                # bbox_all_foot_point: (num_pairs, 6):              (x1, y1, condifence, x2, y2, confidence)
                bbox_foot_point = check_same_person(bbox_all_foot_point, ground_truth)
                if bbox_foot_point.shape[0] == 0:    # no same person, we can just continue
                    continue
                loss, tracking_loss, foot_point_loss = criterion(tracking_pred, bbox_foot_point, homo_mat, ground_truth)     # shape of foot_point: (num_same_persons, 2, 6)
                
            else: # args.bbox == False  
                input = input_orig
                tracking_pred, homo_mat = model(input)       # transform from float to double
                tracking_pred = tracking_pred.squeeze(0)                #(1, num_pairs, 2) -> (num_pairs, 2)
                homo_mat = modify_homo_mat(homo_mat)
                all_foot_point = input[:, :, 15:17,:].squeeze(0)                      #(1, num_pairs, 17, 6) ->(num_pairs, 2, 6)
                foot_point = check_same_person(all_foot_point, ground_truth)
                if foot_point.shape[0] == 0:    # no same person, we can just continue
                    continue
                loss, tracking_loss, foot_point_loss = criterion(tracking_pred, foot_point, homo_mat, ground_truth)     # shape of foot_point: (num_same_persons, 2, 6)
                
            print("epoch: {}, frame: {}, loss: {}".format(epoch, idx, loss))
            epoch_total_loss += loss
            epoch_total_fploss += foot_point_loss
            epoch_total_tkloss += tracking_loss
            # backward propagation
            optimizer.zero_grad()
            #print("hsy before loss\n")
            loss.backward()
            optimizer.step()
            enum_times += 1
            if args.eva < 1 and (idx+1) % eva_frame == 0:
                utils.evaluate(model, logger_file, epoch, args)
            #if idx >= 1500:  # only train with ### pictures
            #    enum_times = idx + 1
            #     break
            #break
        #break
        epoch_avg_loss.append(epoch_total_loss.cpu().detach().numpy() / enum_times)
        epoch_avg_fploss.append(epoch_total_fploss.cpu().detach().numpy() / enum_times)
        epoch_avg_tkloss.append(epoch_total_tkloss.cpu().detach().numpy() / enum_times)
        if args.eva >= 1 and (epoch+1) % args.eva == 0:
            utils.evaluate(model, logger_file, epoch, args)
    print("average total loss in each epoch is {}".format(epoch_avg_loss), file=logger_file)
    print("average footpoint loss in each epoch is {}".format(epoch_avg_fploss), file=logger_file)
    print("average tracking loss in each epoch is {}".format(epoch_avg_tkloss), file=logger_file)
    #pdb.set_trace()
    end_time = time.time()
    print("total time: {}".format(end_time - start_time), file = logger_file)
    logger_file.close()