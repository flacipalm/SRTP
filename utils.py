import torch
from dataset import KeyDataset
from torch.utils.data.dataloader import DataLoader
from visualize_2d.dataset_loader import *
import numpy as np
import scipy.optimize
import time
from sklearn.metrics import roc_auc_score
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:1,2,3')
def check_same_person(all_foot_point, ground_truth):
    """
        Correctness checked: yes
        Discription:
        Args:
            all_foot_point (tensor): (num_pairs, 2, 6)
            ground_truth (tensor): (1, num_pairs)
        Return:
            res (tensor): (num_same_persons, 2, 6)
    """
    ground_truth = ground_truth.squeeze(0)
    first_time = 1
    res = torch.zeros(0)   # init
    for idx, judge in enumerate(ground_truth):
        if judge == 1 and first_time == 1:
            res = all_foot_point[idx].unsqueeze(0)
            first_time = 0
        elif judge == 1 and first_time != 1:
            res = torch.cat((res, all_foot_point[idx].unsqueeze(0)), 0)
    return res

def modify_homo_mat(homo_mat):
    """
        Correctness checked: yes, but don't know whether it's precisely true
        Discription: construct homography matrix
        Args:
            homo_mat(tensor): (1, 8)
        Return:
            res (tensor): (3, 3)
    """
    homo_mat = homo_mat.squeeze(0)       #(1, 8) -> (8)
    one = torch.ones(1,device=device)
    res = torch.cat((homo_mat,one)).reshape(3,3)
    return res

def eva_matching(tracking_pred, ground_truth):
    '''
    Discription: evaluate matching performance on two views and on one frame
    Args:
        tracking_pred(tensor）: (num_pairs, 2)
        ground_truth (tensor): (1, num_pairs)
    Return:
        correct_num_pairs: int
        num_pairs: int
    '''
    num_pairs = tracking_pred.shape[0]  # number of pairs for this frame (two views combination)
    ground_truth_copy = ground_truth.squeeze(0)
    correct_num_pairs = 0
    true_negative = 0
    true_positive = 0
    num_pairs_0 = 0
    num_pairs_1 = 0
    for idx, gt in enumerate(ground_truth_copy):
        if tracking_pred[idx][0] > tracking_pred[idx][1]:
            pred_pair = 0   # the prediction score of this pair. 0 for not matched, 1 for matched.
        else:
            pred_pair = 1
        if gt == 0:
            num_pairs_0 += 1
        else:   # gt == 1
            num_pairs_1 += 1
        if pred_pair == 0 and gt == 0:  # negative sample prediction correctly
            true_negative += 1
        elif pred_pair == 1 and gt == 1:   # positive sample prediction correctly
            true_positive += 1
    false_positive = num_pairs_0 - true_negative
    false_negative = num_pairs_1 - true_positive
    return true_negative, true_positive, false_negative, false_positive

def eva_matching_hungarian(tracking_pred, ground_truth, pair_shape):
    '''
    Discription: evaluate matching performance on two views and on one frame
    Args:
        tracking_pred(tensor）: (num_pairs, 2)
        ground_truth (tensor): (1, num_pairs)
        pair_shape(tensor): (1, num_view1, num_view2)
    Return:
        correct_num_pairs: int
        num_pairs: int
    '''
    pair_shape = pair_shape[0]
    num_pairs = tracking_pred.shape[0]  # number of pairs for this frame (two views combination)
    ground_truth_copy = ground_truth.squeeze(0) #（num_pairs）
    correct_num_pairs = 0
    true_negative = 0
    true_positive = 0
    num_pairs_0 = 0
    num_pairs_1 = 0
    #pdb.set_trace()
    tracking_pred_matrix = torch.reshape(tracking_pred[:,1], (int(pair_shape[0].item()), int(pair_shape[1].item()))).detach().cpu().numpy()       
    row, col = scipy.optimize.linear_sum_assignment(tracking_pred_matrix, maximize=True)        # numpy
    pred_matrix = np.zeros_like(tracking_pred_matrix)
    for i in range(len(row)):
        pred_matrix[row[i]][col[i]] = 1
    pred_flat = np.reshape(pred_matrix, (num_pairs))
    #pdb.set_trace()
    for idx, gt in enumerate(ground_truth_copy):
        pred_pair = pred_flat[idx]
        '''
        if tracking_pred[idx][0] > tracking_pred[idx][1]:
            pred_pair = 0   # the prediction of this pair. 0 for not matched, 1 for matched.
        else:
            pred_pair = 1
        '''
        if gt == 0:
            num_pairs_0 += 1
        else:   # gt == 1
            num_pairs_1 += 1
        if pred_pair == 0 and gt == 0:  # negative sample prediction correctly
            true_negative += 1
        elif pred_pair == 1 and gt == 1:   # positive sample prediction correctly
            true_positive += 1
    false_positive = num_pairs_0 - true_negative
    false_negative = num_pairs_1 - true_positive
    return true_negative, true_positive, false_negative, false_positive

def evaluate(model, logger_file, epoch, args):
    '''
    Discription: evaluate the matching accuracy rate of our trained model.
    '''
    testset = KeyDataset(scene=args.scene, view=args.view, Test=True, Bbox=args.bbox, Hungarian=args.hungarian)
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True, num_workers=0)
    total_correct_num_pairs = 0
    total_num_pairs = 0
    total_TN = 0
    total_TP = 0
    total_FN = 0
    total_FP = 0
    total_num_pairs_0 = 0
    total_num_pairs_1 = 0
    precision = 0
    recall = 0
    tracking_pred_all = []     # create a sentinel, need to be deleted later
    ground_truth_all = []
    for idx, item in enumerate(test_loader):
        input, ground_truth, pair_shape = item
        tracking_pred, _ = model(input)       # transform from float to double
        tracking_pred = tracking_pred.squeeze(0)                #(1, num_pairs, 2) -> (num_pairs, 2)
        pred_score = tracking_pred[:,1].cpu().detach().numpy().tolist()
        ground_truth_copy = ground_truth.squeeze(0).cpu().detach().numpy().tolist()

        try:
            for i in range(len(pred_score)):
                tracking_pred_all.append(pred_score[i])
                ground_truth_all.append(ground_truth_copy[i])
        except: 
            pdb.set_trace()

        if args.hungarian == False:
            true_negative, true_positive, false_negative, false_positive = eva_matching(tracking_pred, ground_truth)
        else: # use hungarian
            true_negative, true_positive, false_negative, false_positive = eva_matching_hungarian(tracking_pred, ground_truth, pair_shape)
        
        total_correct_num_pairs += true_negative + true_positive
        total_num_pairs += true_negative + true_positive + false_negative + false_positive
        
        total_TN += true_negative
        total_TP += true_positive
        total_FN += false_negative
        total_FP += false_positive

        total_num_pairs_0 += false_positive + true_negative
        total_num_pairs_1 += false_positive + true_negative
        total_acc_rate = total_correct_num_pairs / total_num_pairs        # may need to transer to double
        try:
            negative_sample_acc_rate = total_TN / (total_TN + total_FP)
        except:
            if total_num_pairs_0 == 0:
                negative_sample_acc_rate = 0
        try:
            positive_sample_acc_rate = total_TP / (total_TP + total_FN)
        except:
            if total_num_pairs_1 == 1:
                positive_sample_acc_rate = 0

        if idx % 1000 == 0 and idx != 0: # print out the evaluate condition for every ### batches
            try:
                precision = total_TP / (total_TP + total_FP)
            except:
                precision = 0
            try:
                recall = total_TP / (total_TP + total_FN)
            except:
                recall = 0
            try:
                f05_score = 1.25 * (precision * recall) / (0.25 * precision + recall)
            except:
                f05_score = 0
            try:
                f1_score = 2 * (precision * recall) / (precision + recall)
            except:
                f1_score = 0
            try:
                f2_score = 5 * (precision * recall) / (4 * precision + recall)
            except:
                f2_score = 0

            print("batch：NO.{}, total correct prediction: {},  total input pairs: {}, Accuracy: {}.".format(idx, total_correct_num_pairs, total_num_pairs, total_acc_rate),"\n",
                  "positive samples number: {}, true positive rate (Recall): {}.".format(total_num_pairs_1, positive_sample_acc_rate),"\n",
                  "negative samples number: {}, true negative rate (specificity): {}.".format(total_num_pairs_0, negative_sample_acc_rate),"\n",
                  "precision: {}, recall:{}, F0.5 score:{}, F1 score:{}, F2 score:{}".format(precision, recall, f05_score, f1_score, f2_score))

        # if idx >= 1500:  #evaluate for only ### pictures
        #    break
    # print("Evaluation is over, the average accuracy rate of matching is {}", acc_rate)

    ground_truth_all = np.array(ground_truth_all)
    tracking_pred_all = np.array(tracking_pred_all)
    try:
        precision = total_TP / (total_TP + total_FP)
    except:
        precision = 0
    try:
        recall = total_TP / (total_TP + total_FN)
    except:
        recall = 0
    try:
        type_one_err = total_FP / (total_FP + total_TN)
    except:
        type_one_err = 0
    try:
        type_two_err = total_FN / (total_TP + total_FN)
    except:
        type_two_err = 0
    try:
        f05_score = 1.25 * (precision * recall) / (0.25 * precision + recall)
    except:
        f05_score = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except:
        f1_score = 0
    try:
        f2_score = 5 * (precision * recall) / (4 * precision + recall)
    except:
        f2_score = 0
    try:
        mcc = (total_TP * total_TN - total_FP * total_FN) / np.sqrt((total_TP + total_FP) * (total_TP + total_FN) * (total_TN + total_FP) * (total_TN + total_FN)) 
    except: 
        mcc = 0
    roc_auc = roc_auc_score(ground_truth_all, tracking_pred_all)
    print("it's evaluation after epoch：{}, total correct prediction: {},  total input pairs: {}, Accuracy: {}.".format(epoch, total_correct_num_pairs, total_num_pairs, total_acc_rate),"\n",
          "positive samples number: {}, true positive rate (Recall): {}.".format(total_num_pairs_1, positive_sample_acc_rate),"\n",
          "negative samples number: {}, true negative rate (specificity): {}.".format(total_num_pairs_0, negative_sample_acc_rate),"\n",
          "precision: {}, recall: {}, F0.5 score: {}, F1 score: {}, F2 score: {}".format(precision, recall, f05_score, f1_score, f2_score), "\n",
          "ROC AUC score: {}, MCC score: {}".format(roc_auc, mcc), "\n",
          "Type-I error (FPR): {}, Type-II error (FNR): {}".format(type_one_err, type_two_err), "\n", file = logger_file)

def evaluate_new(model, logger_file, epoch, args):
    '''
    Discription: evaluate the matching accuracy rate of our trained model.
    '''
    sample_scene=[]
    for scene in args.scene:
      sample_scene.append('sample'+ str(args.scene[scene]))
    sample = multi_human36m_syn_data(sample_name=sample_scene)
    train_loader = DataLoader(dataset=sample, batch_size=1, shuffle=args.shuffle, num_workers=0)
    #testset = KeyDataset(scene=args.scene, view=args.view, Test=True)
    #test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=True, num_workers=0)
    total_correct_num_pairs = 0
    total_num_pairs = 0
    total_TN = 0
    total_TP = 0
    total_FN = 0
    total_FP = 0
    total_num_pairs_0 = 0
    total_num_pairs_1 = 0

    for idx, (frame, pairs, match) in enumerate(train_loader):
        input = pairs
        ground_truth = match.squeeze(0)  # （1,9） -> (9)
        tracking_pred, homo_mat = model(input)  # transform from float to double
        tracking_pred = tracking_pred.squeeze(0)  # (1, num_pairs, 2) -> (num_pairs, 2)
        #input, ground_truth = item
        #tracking_pred, _ = model(input)  # transform from float to double
        tracking_pred = tracking_pred.squeeze(0)  # (1, num_pairs, 2) -> (num_pairs, 2)
        true_negative, true_positive, false_negative, false_positive = eva_matching(tracking_pred, ground_truth)

        total_correct_num_pairs += true_negative + true_positive
        total_num_pairs += true_negative + true_positive + false_negative + false_positive
        
        total_TN += true_negative
        total_TP += true_positive
        total_FN += false_negative
        total_FP += false_positive

        total_num_pairs_0 += false_positive + true_negative
        total_num_pairs_1 += false_positive + true_negative

        total_acc_rate = total_correct_num_pairs / total_num_pairs  # may need to transer to double
        try:
            negative_sample_acc_rate = total_TN / (total_TN + total_FP)
        except:
            if total_num_pairs_0 == 0:
                negative_sample_acc_rate = 0
        try:
            positive_sample_acc_rate = total_TP / (total_TP + total_FN)
        except:
            if total_num_pairs_1 == 1:
                positive_sample_acc_rate = 0

        if idx % 1000 == 0 and idx != 0: # print out the evaluate condition for every ### batches
            try:
                precision = total_TP / (total_TP + total_FP)
            except:
                precision = 0
            try:
                recall = total_TP / (total_TP + total_FN)
            except:
                recall = 0
            try:
                f05_score = 1.25 * (precision * recall) / (0.25 * precision + recall)
            except:
                f05_score = 0
            try:
                f1_score = 2 * (precision * recall) / (precision + recall)
            except:
                f1_score = 0
            try:
                f2_score = 5 * (precision * recall) / (4 * precision + recall)
            except:
                f2_score = 0

            print("batch：NO.{}, total correct prediction: {},  total input pairs: {}, Accuracy: {}.".format(idx, total_correct_num_pairs, total_num_pairs, total_acc_rate),"\n",
                  "positive samples number: {}, true positive rate (Recall): {}.".format(total_num_pairs_1, positive_sample_acc_rate),"\n",
                  "negative samples number: {}, true negative rate (specificity): {}.".format(total_num_pairs_0, negative_sample_acc_rate),"\n",
                  "precision: {}, recall:{}, F0.5 score:{}, F1 score:{}, F2 score:{}".format(precision, recall, f05_score, f1_score, f2_score))

        # if idx >= 1500:  #evaluate for only ### pictures
        #    break
    # print("Evaluation is over, the average accuracy rate of matching is {}", acc_rate)
    ground_truth_all = np.array(ground_truth_all)
    tracking_pred_all = np.array(tracking_pred_all)
    try:
        precision = total_TP / (total_TP + total_FP)
    except:
        precision = 0
    try:
        recall = total_TP / (total_TP + total_FN)
    except:
        recall = 0
    try:
        type_one_err = total_FP / (total_FP + total_TN)
    except:
        type_one_err = 0
    try:
        type_two_err = total_FN / (total_TP + total_FN)
    except:
        type_two_err = 0
    try:
        f05_score = 1.25 * (precision * recall) / (0.25 * precision + recall)
    except:
        f05_score = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except:
        f1_score = 0
    try:
        f2_score = 5 * (precision * recall) / (4 * precision + recall)
    except:
        f2_score = 0
    try:
        mcc = (total_TP * total_TN - total_FP * total_FN) / np.sqrt((total_TP + total_FP) * (total_TP + total_FN) * (total_TN + total_FP) * (total_TN + total_FN)) 
    except: 
        mcc = 0
    roc_auc = roc_auc_score(ground_truth_all, tracking_pred_all)
    print("it's evaluation after epoch：{}, total correct prediction: {},  total input pairs: {}, Accuracy: {}.".format(epoch, total_correct_num_pairs, total_num_pairs, total_acc_rate),"\n",
          "positive samples number: {}, true positive rate (Recall): {}.".format(total_num_pairs_1, positive_sample_acc_rate),"\n",
          "negative samples number: {}, true negative rate (specificity): {}.".format(total_num_pairs_0, negative_sample_acc_rate),"\n",
          "precision: {}, recall: {}, F0.5 score: {}, F1 score: {}, F2 score: {}".format(precision, recall, f05_score, f1_score, f2_score), "\n",
          "ROC AUC score: {}, MCC score: {}".format(roc_auc, mcc), "\n",
          "Type-I error (FPR): {}, Type-II error (FNR): {}".format(type_one_err, type_two_err), "\n", file = logger_file)