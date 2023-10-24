# loss function
import torch.nn as nn
import torch
import pdb
import scipy.optimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:1,2,3')
class LossFunc(nn.Module):
    def __init__(self, Bbox=False, Loss_comb=0):
        super(LossFunc, self).__init__()
        self.bbox = Bbox
        self.loss_comb = Loss_comb
        #self.hungarian = Hungarian
        return

    def forward(self, tracking_pred, foot_point, homo_mat, ground_truth):
        '''
            Correctness checked: yes
            Discription: define the loss function which combines tracking_loss(computed by CrossEntropyLoss between tracking prediction and groundtruth)
                         and foot_point_loss(computed by 2 norm between foot points in two different views, using homography matrix as transformation)
            Args:
                tracking_pred (tensor）: (num_pairs, 2)
                foot_point(tensor): if Bbox = False, (num_same_persons, 2, 6)
                                  : if Bbox = True, (num_same_persons, 6)
                homo_mat(tensor): (3, 3)
                ground_truth (tensor): (1, num_pairs)
                pair_shape(tensor): (num_view1, num_view2)
            Return:
                loss
        ''' 
        #pair_shape = pair_shape.squeeze(0)
        ground_truth = ground_truth.squeeze(0).long()  # data type of "target" must be long instead of double
        Cross_Entropy_Loss = nn.CrossEntropyLoss()
        tracking_loss = Cross_Entropy_Loss(tracking_pred, ground_truth)
        '''
        if self.hungarian == False:
            Cross_Entropy_Loss = nn.CrossEntropyLoss()
            tracking_loss = Cross_Entropy_Loss(tracking_pred, ground_truth)
        else:   # use hungarian algorithm
            #pdb.set_trace()
            tracking_pred_matrix = torch.reshape(tracking_pred[:,1], (int(pair_shape[0].item()), int(pair_shape[1].item())))         
            row, col = scipy.optimize.linear_sum_assignment(tracking_pred_matrix, maximize=True)
            pdb.set_trace()
        '''
        if self.bbox == False:
            foot_points = foot_point.reshape(foot_point.shape[0], foot_point.shape[1], 2, 3)
            foot_point, _ = foot_points.split(2, dim=-1)#footpoint(n, 2, 2, 2)
            ones = torch.ones((foot_point.shape[0], foot_point.shape[1], foot_point.shape[2], 1), device=device)
            view1_foot, view2_foot = torch.cat((foot_point, ones), dim=-1).split(1, dim=-2)  #view_foot (n,2,1,2)
            homo_mat = homo_mat.repeat(view2_foot.shape[0], view2_foot.shape[1], 1, 1)
            view2 = torch.matmul(homo_mat, view2_foot.squeeze(-2).unsqueeze(-1)).squeeze(-1)
            view2_point = view2 / (view2[..., 2].reshape(view2.size(0), view2.size(1), 1))
            foot_point_loss = torch.mean(torch.norm(view1_foot.squeeze(-2) - view2_point, dim=-2))
        else: 
            foot_points = foot_point.reshape(foot_point.shape[0], 1, 2, 3)
            foot_point, _ = foot_points.split(2, dim=-1)
            ones = torch.ones((foot_point.shape[0], foot_point.shape[1], foot_point.shape[2], 1), device=device)
            view1_foot, view2_foot = torch.cat((foot_point, ones), dim=-1).split(1, dim=-2)  # reshape (x,y,2,3), cut (x,y,2,2) , cat (x,y,2,3)
            homo_mat = homo_mat.repeat(view2_foot.shape[0], view2_foot.shape[1], 1, 1)
            view2 = torch.matmul(homo_mat, view2_foot.squeeze(-2).unsqueeze(-1)).squeeze(-1)
            view2_point = view2 / (view2[..., 2].reshape(view2.size(0), view2.size(1), 1))
            foot_point_loss = torch.mean(torch.norm(view1_foot.squeeze(-2) - view2_point, dim=-2)) 

        if self.loss_comb == 0:
            loss = tracking_loss + foot_point_loss
        elif self.loss_comb == 1:
            loss = foot_point_loss
        else: # self.loss_comb == 2;
            loss = tracking_loss
        # loss = tracking_loss
        # loss = foot_point_loss
        #pdb.set_trace()
        return loss, tracking_loss, foot_point_loss







'''
    def modify_tracking_pred(self, tracking_pred):
        """
            Discription: modify tracking_pred to meet the form of CrossEntropyLoss
            Args:
                tracking_pred (tensor):  (num_pairs)           
            Return:
                res (tensor): (num_pairs, 2)
        """
        pdb.set_trace()
        num_pairs = len(tracking_pred)
        res = torch.zeros(num_pairs, 2)
        for idx, item in enumerate(tracking_pred):
            if item == 0 :
                res[idx][0] = 1
            else:   #item == 1
                res[idx][1] = 1
        return res     
'''