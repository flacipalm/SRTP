import torch
import pdb
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
def subregressor(pose_pred, pose_2d):
    """create a linear regression in single person with frame width w

    Args:
        pose_pred (tensor): predicted_3d_pose with shape[joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[joint,3]
        update (bool): if compute loss
    """
    # initial the matrix
    ones = torch.ones([17,1], device=device)
    A = torch.cat((pose_pred, ones), dim=1)  
    B = torch.cat((pose_2d, ones), dim=1) 
    
    # get component [128,17]
    try:
      T = torch.mm(torch.mm(torch.inverse(torch.mm(A.T,A)),A.T),B)
    except:
      return torch.tensor([float('inf')],device=device)
    A_ = torch.mm(A,T)
    A2,_ = torch.split(A_, (3,1), dim=1)
    B2,_ = torch.split(B, (3,1), dim=1)
    dis = torch.mean(torch.norm(A2-B2,dim=1))
    return dis
    

def pair(pose_pred, pose_2d):
    """
    Args:
        pose_pred (tensor): predicted_3d_pose with shape[n,joint,3]
        pose_2d (tensor): pixel_2d_pose with shape[m,joint,3]
    """
    m = pose_pred.shape[0]
    n = pose_2d.shape[0]
    dis = torch.zeros([m,n], device=device)
    for i in range(m):
        for j in range(n):
            dis[i,j] = subregressor(pose_pred[i], pose_2d[j])
    return dis

def accuracy(dis, gth):
    """
    Args:
        dis (tensor): [m,n]
        gth (tensor): [m,n]
    return:
        acu(list): [# positive, # negative, #pT, #NT]
    """
    pred = Hungearian(dis)
    True_P = pred*gth
    
    acu = [sum(gth), m*n-sum(gth), sum(True_P), m*n-sum(gth)-sum(pred)+sum(True_P)]
    return  acu

def base_line(pose_pred):
    """
    Args:
        pose_pred (dic): predicted_3d_pose with shape{frame, view tensor[n,17,3]}
    """
    accu = [0,0,0,0]# # positive, # negative, #pT, #NT
    for i in range (len(pose_pred)):
       for j in range (3):
             dis = pair(pose_pred[i]['view2'], pose_pred[i]['view2'-1]) 
             accu_temp = accuracy(dis, gth)
             accu = accu + accu_temp
    return accu
                


        
     

pose_pred = torch.ones([3,17,3], device = device)
pose_2d = torch.ones([4,17,3], device = device)

T = pair(pose_pred, pose_2d)

print(T)
