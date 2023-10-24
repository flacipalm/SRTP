import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch

Tensor = torch.tensor
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
import argparse

#from visualization_multi_2d import render_animation
from visualize_2d.dataset_loader import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:3')
def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    parser.add_argument('--scene', default=[0,1,2,3,4,5,6,7,8,9], type=list,
                        help='scene to trainset')
    parser.add_argument('--test_scene', default=[0,1,2,3,4,5,6,7,8,9], type=list,
                        help='scene to testset')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle or not')
    parser.add_argument('--epoch', default=50, type=int,
                        help='training epoch')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--eva', default=1, type=float,
                        help='make evaluation after each # epoch')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    t = time.localtime()
    start_time = time.time()
    args = parse_args()
    print(args)

    if t.tm_hour >= 14:  # change from Beijing's time to Chicago's time
        hour_mod = t.tm_hour - 14
        day_mod = t.tm_mday
    else:
        hour_mod = t.tm_hour + 10
        day_mod = t.tm_mday - 1
    logger_file = open(
        './logger_after0414/{}{}{}_{}:{}:{}_train:{}_test:{}_lr:{}_eva:{}_epoch:{}_newdata.txt'.format(t.tm_year, t.tm_mon, day_mod,
                                                                                         hour_mod, t.tm_min, t.tm_sec,
                                                                                         args.scene, args.test_scene, args.lr,
                                                                                         args.eva, args.epoch), 'w')
    # pdb.set_trace()

    model = TrackingMatch().to(device).double()
    criterion = LossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    epochs = args.epoch
    # pdb.set_trace()
    epoch_avg_loss = []
    epoch_avg_fploss = []  # foot_point_loss
    epoch_avg_tkloss = []  # tracking_loss
    sample_scene=[]
    for scene in args.scene:
        scene = int(scene)
        sample_scene.append('sample'+ str(args.scene[scene]))
    sample = multi_human36m_syn_data(sample_name=sample_scene)
    train_loader = DataLoader(dataset=sample, batch_size=1, shuffle=args.shuffle, num_workers=0)

    if args.eva < 1:
        eva_frame = args.eva * len(train_loader)
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_total_fploss = 0
        epoch_total_tkloss = 0
        enum_times = 0
        for index, (frame, pairs, match) in enumerate(train_loader):
            # forward propagation
            # shape of input: (1, num_pairs, 17, 6)
            # shape of ground_truth: (num_pairs)

            # pairs:(1,9,17,6)
            # match:(1,9)
            #input, ground_truth = item
            input = pairs
            ground_truth = match.squeeze(0)     #（1,9） -> (9)
            tracking_pred, homo_mat = model(input)  # transform from float to double
            # pdb.set_trace()
            tracking_pred = tracking_pred.squeeze(0)  # (1, num_pairs, 2) -> (num_pairs, 2)
            homo_mat = modify_homo_mat(homo_mat)
            left_foot = input[:, :, 3, :].unsqueeze(2).squeeze(0)
            right_foot = input[:, :, 6, :].unsqueeze(2).squeeze(0)
            #db.set_trace()
            all_foot_point = torch.cat((left_foot,right_foot), 1)
            #all_foot_point = input[:, :, 15:17, :].squeeze(0)  # (1, num_pairs, 17, 6) ->(num_pairs, 2, 6)          ##
            foot_point = check_same_person(all_foot_point, ground_truth)
            if foot_point.shape[0] == 0:  # no same person, we can just continue
                continue
            loss, tracking_loss, foot_point_loss = criterion(tracking_pred, foot_point, homo_mat,
                                                             ground_truth)  # shape of foot_point: (num_same_persons, 2, 6)
            print("epoch: {}, frame: {}, loss: {}".format(epoch, frame, loss))
            epoch_total_loss += loss
            epoch_total_fploss += foot_point_loss
            epoch_total_tkloss += tracking_loss
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            enum_times += 1
            if args.eva < 1 and (index + 1) % eva_frame == 0:
                utils.evaluate(model, logger_file, epoch, args)
            # if idx >= 1500:  # only train with ### pictures
            #    enum_times = idx + 1
            #     break
            # break
        # break
        epoch_avg_loss.append(epoch_total_loss.cpu().detach().numpy() / enum_times)
        epoch_avg_fploss.append(epoch_total_fploss.cpu().detach().numpy() / enum_times)
        epoch_avg_tkloss.append(epoch_total_tkloss.cpu().detach().numpy() / enum_times)
        if args.eva >= 1 and (epoch + 1) % args.eva == 0:
            utils.evaluate_new(model, logger_file, epoch, args)
    print("average total loss in each epoch is {}".format(epoch_avg_loss), file=logger_file)
    print("average footpoint loss in each epoch is {}".format(epoch_avg_fploss), file=logger_file)
    print("average tracking loss in each epoch is {}".format(epoch_avg_tkloss), file=logger_file)
    # pdb.set_trace()
    end_time = time.time()
    print("total time: {}".format(end_time - start_time), file=logger_file)
    logger_file.close()