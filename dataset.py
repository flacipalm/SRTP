import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch 
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:1,2,3')
class KeyDataset(Dataset):
    def __init__(self, scene, view, Test=False, Bbox=False, Hungarian=False):
        # data shape in .npz file: (frame, id, 17, 3), but existing people are all closely arranged while nonexisting people are all arranged as 0 after
        # need to output (1, num_pairs, 17, 6)
        # data1(numpy), data2(numpy): (frame, id, 17, 3)
        # data1_gt(numpy), data2_gt(numpy): (id)
        # data1_bbox, data2_bbox: (frame, id, 3, 3)
        self.Bbox = Bbox
        self.Hungarian = Hungarian
        self.data1, self.data2, self.data1_gt, self.data2_gt, self.data1_bbox, self.data2_bbox = self.import_data(Test, scene, view)
        self.Test = Test
        #pdb.set_trace()

    def __getitem__(self, frame_idx):
        gt_data = self.modify_gt(self.data1_gt[frame_idx], self.data2_gt[frame_idx])
        if self.Bbox == False:       # input keypoints into network
            Mat1 = self.wrap_del_tensor_0(self.data1[frame_idx])
            Mat2 = self.wrap_del_tensor_0(self.data2[frame_idx])
        else:
            Mat1 = self.wrap_del_tensor_0(self.data1_bbox[frame_idx])
            Mat2 = self.wrap_del_tensor_0(self.data2_bbox[frame_idx]) 
        data = self.dim_transform(Mat1, Mat2)
        '''
        if self.Hungarian == False:
            data_mod, gt_mod = self.samples_balance(data, gt_data)
        else: # use hungarian algorithm
            data_mod = data
            gt_mod = gt_data
        '''
        
        pair_shape = torch.Tensor([Mat1.shape[0], Mat2.shape[0]])
        if self.Test == True and self.Hungarian == True:    # skip sample_balance
            data_mod = data
            gt_mod = gt_data 
        else:
            data_mod, gt_mod = self.samples_balance(data, gt_data)
        #pdb.set_trace()
        #if self.Test == True and self.Hungarian == True:
            #pdb.set_trace()
        return torch.from_numpy(data_mod).double().to(device), torch.from_numpy(gt_mod).double().to(device), pair_shape

    def __len__(self):
        return len(self.data1_gt)
    
    def import_data(self, Test, scene, view):
        view = list(map(int, view))
        scene = list(map(int, scene))
        scenes = ['circleRegion', "southGate", "innerShop","movingView",'park',"playground","shopFrontGate","shopSecondFloor","shopSideGate","shopSideSquare"]
        scenes = [scenes[i] for i in scene]
        views = ["Drone", "View1", "View2"]
        views = [views[i] for i in view]
        #pdb.set_trace()
        data_view1 = np.zeros((1, 1, 17, 3)) #[frame, id, 17,3]
        data_view2 = np.zeros((1, 1, 17, 3))
        
        data_view1_gt = np.zeros((1, 1)) #[frame, id, 17,3]
        data_view2_gt = np.zeros((1, 1))

        data_view1_bbox = np.zeros((1, 1, 3, 3)) #[frame, id, 3, 3]
        data_view2_bbox = np.zeros((1, 1, 3, 3)) 
        for scene in scenes:
            for i in range(len(views)): # only depends on len, have problem but could satisfy our requirement
                data1_i = np.load('/home/xqr/fzy/0312/{}_{}_20220312.npz'.format(scene, views[i]))
                data2_i = np.load('/home/xqr/fzy/0312/{}_{}_20220312.npz'.format(scene, views[i-1]))
                data1 = data1_i["keypoints"][1:, :, :, :]
                data2 = data2_i["keypoints"][1:, :, :, :]  # start from frame 1
                data1_gt = data1_i["id_val"][1:, :]  # shape: (frame, id)    the matrix stores the true id information
                data2_gt = data2_i["id_val"][1:, :]
                data1_bbox = data1_i["bbox"][1:, :, :, :]
                data2_bbox = data2_i["bbox"][1:, :, :, :]
                #pdb.set_trace()

                # decreases circleRegion drone, view2 by 10 frames
                # process for circleRegion since in this scene, drone and view2 has 3201 frames, and view1 only has 3191 frames due to incomplete tracklet.
                if scene == "circleRegion":
                    if views[i] == 'Drone':
                        data1 = data1[:-10, :, :, :]    # drone
                        data2 = data2[:-10, :, :, :]    # view2
                        data1_gt = data1_gt[:-10, :]
                        data2_gt = data2_gt[:-10, :]
                        data1_bbox = data1_bbox[:-10, :, :, :]
                        data2_bbox = data2_bbox[:-10, :, :, :]
                    elif views[i] == 'View1':
                        data2 = data2[:-10, :, :, :]    # drone
                        data2_gt = data2_gt[:-10, :]
                        data2_bbox = data2_bbox[:-10, :, :, :]
                    elif views[i] == 'View2':
                        data1 = data1[:-10, :, :, :]    # view2
                        data1_gt = data1_gt[:-10, :]
                        data1_bbox = data1_bbox[:-10, :, :, :]
                #pdb.set_trace()

                # normalization of the dataset
                # notice that resolution of drone and view2 is (1920, 1080), and the resolution of view1 is (3840, 2048)
                if views[i] == 'Drone':     # data1 is drone, data2 is view2
                    data1[:, :, :, 0] = (data1[:, :, :, 0] / 960) - 1
                    data1[:, :, :, 1] = (data1[:, :, :, 1] / 540) - 1
                    data2[:, :, :, 0] = (data2[:, :, :, 0] / 960) - 1
                    data2[:, :, :, 1] = (data2[:, :, :, 1] / 540) - 1
                    data1_bbox[:, :, :, 0] = (data1_bbox[:, :, :, 0] / 960) - 1
                    data1_bbox[:, :, :, 1] = (data1_bbox[:, :, :, 1] / 540) - 1
                    data2_bbox[:, :, :, 0] = (data2_bbox[:, :, :, 0] / 960) - 1
                    data2_bbox[:, :, :, 1] = (data2_bbox[:, :, :, 1] / 540) - 1

                elif views[i] == 'View1':   # data1 is view1, data2 is drone
                    data1[:, :, :, 0] = (data1[:, :, :, 0] / 1920) - 1
                    data1[:, :, :, 1] = (data1[:, :, :, 1] / 1024) - 1
                    data2[:, :, :, 0] = (data2[:, :, :, 0] / 960) - 1
                    data2[:, :, :, 1] = (data2[:, :, :, 1] / 540) - 1
                    data1_bbox[:, :, :, 0] = (data1_bbox[:, :, :, 0] / 1920) - 1
                    data1_bbox[:, :, :, 1] = (data1_bbox[:, :, :, 1] / 1024) - 1
                    data2_bbox[:, :, :, 0] = (data2_bbox[:, :, :, 0] / 960) - 1
                    data2_bbox[:, :, :, 1] = (data2_bbox[:, :, :, 1] / 540) - 1
                elif views[i] == 'View2':   # data1 is view2, data2 is view1
                    data1[:, :, :, 0] = (data1[:, :, :, 0] / 960) - 1
                    data1[:, :, :, 1] = (data1[:, :, :, 1] / 540) - 1
                    data2[:, :, :, 0] = (data2[:, :, :, 0] / 1920) - 1
                    data2[:, :, :, 1] = (data2[:, :, :, 1] / 1024) - 1
                    data1_bbox[:, :, :, 0] = (data1_bbox[:, :, :, 0] / 960) - 1
                    data1_bbox[:, :, :, 1] = (data1_bbox[:, :, :, 1] / 540) - 1
                    data2_bbox[:, :, :, 0] = (data2_bbox[:, :, :, 0] / 1920) - 1
                    data2_bbox[:, :, :, 1] = (data2_bbox[:, :, :, 1] / 1024) - 1
                
                # first half for training and second half for testing
                if (Test == True):
                    _, data1 = np.array_split(data1, 2, axis=0)
                    _, data2 = np.array_split(data2, 2, axis=0)
                    _, data1_gt = np.array_split(data1_gt, 2, axis=0)
                    _, data2_gt = np.array_split(data2_gt, 2, axis=0)
                    _, data1_bbox = np.array_split(data1_bbox, 2, axis=0)
                    _, data2_bbox = np.array_split(data2_bbox, 2, axis=0)
                else:
                    data1, _ = np.array_split(data1, 2, axis=0)
                    data2, _ = np.array_split(data2, 2, axis=0) 
                    data1_gt, _ = np.array_split(data1_gt, 2, axis=0)
                    data2_gt, _ = np.array_split(data2_gt, 2, axis=0)
                    data1_bbox, _ = np.array_split(data1_bbox, 2, axis=0)
                    data2_bbox, _ = np.array_split(data2_bbox, 2, axis=0) 
                #padding to have same size

                if (data_view1.shape[1] >= data1.shape[1]):
                    n = data_view1.shape[1] - data1.shape[1]
                    data1 = np.pad(data1, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                    data1_gt = np.pad(data1_gt, ((0, 0), (0, n)), 'constant')
                    data1_bbox = np.pad(data1_bbox, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                else:
                    n = -data_view1.shape[1] + data1.shape[1]
                    data_view1 = np.pad(data_view1, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                    data_view1_gt = np.pad(data_view1_gt, ((0, 0), (0, n)), 'constant')
                    data_view1_bbox = np.pad(data_view1_bbox, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                if (data_view2.shape[1] >= data2.shape[1]):
                    n = data_view2.shape[1] - data2.shape[1]
                    data2 = np.pad(data2, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                    data2_gt = np.pad(data2_gt, ((0, 0), (0, n)), 'constant')
                    data2_bbox = np.pad(data2_bbox, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                else:
                    n = -data_view2.shape[1] + data2.shape[1]
                    data_view2 = np.pad(data_view2, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                    data_view2_gt = np.pad(data_view2_gt, ((0, 0), (0, n)), 'constant')
                    data_view2_bbox = np.pad(data_view2_bbox, ((0, 0), (0, n), (0, 0), (0, 0)), 'constant')
                
                data_view1 = np.concatenate((data_view1, data1))
                data_view2 = np.concatenate((data_view2, data2))
                data_view1_gt = np.concatenate((data_view1_gt, data1_gt))
                data_view2_gt = np.concatenate((data_view2_gt, data2_gt))
                data_view1_bbox = np.concatenate((data_view1_bbox, data1_bbox))
                data_view2_bbox = np.concatenate((data_view2_bbox, data2_bbox))

                #print(data_view1.shape)
                #print(data_view2.shape)
        #pdb.set_trace()
        return data_view1[1:,:,:,:], data_view2[1:,:,:,:], data_view1_gt[1:, :], data_view2_gt[1:, :], data_view1_bbox[1:,:,:,:], data_view2_bbox[1:,:,:,:]
        # is_view1
        # 0: data1 is view1, 1: data2 is view1, 2: no view output is view1
    #def balance_neg_pos(data1, data2, data1_g, data2_gt):
        
    
    
    def dim_transform(self, Mat1, Mat2):
        """
            Correctness checked: yes
            Discription:
            Args:
                Mat1 (numpy): (num_person1, n, 3)
                Mat2 (numpy): (num_person2, n, 3)
            Return:
                res  (numpy): (num_pairs, n, 6)
        """
        num_pairs = len(Mat1) * len(Mat2)
        res = np.zeros(num_pairs)
        for idx1, id1 in enumerate(Mat1):
            for idx2, id2 in enumerate(Mat2):
                if idx1 == 0 and idx2 == 0:
                    res = np.expand_dims(np.concatenate((id1, id2), axis=1), axis=0)
                else:
                    item = np.expand_dims(np.concatenate((id1, id2), axis=1), axis=0)
                    res = np.concatenate((res, item), axis=0)
        return res

    def del_tensor_0(self, Mat):
        """
            Correctness checked: yes
            Discription: Get valid pairs
            Args:
                Mat (tensor): id# * n *3.
            Return:
                Mat (tensor): id(real num)* n *3
        """
        #pdb.set_trace()
        ix1 = torch.all(((Mat[..., 0] == -1) & (Mat[..., 1] == -1) & (Mat[..., 2] == 0)), axis=-1)
        ix2 = torch.all(((Mat[..., 0] == 0) & (Mat[..., 1] == 0) & (Mat[..., 2] == 0)), axis=-1)
        ix = ix1 | ix2
        index = []
        # pdb.set_trace()
        for i in range(ix.shape[0]):
            if not ix[i].item():
                index.append(i)
        index = torch.tensor(index)
        Mat = torch.index_select(Mat, 0, index)
        #pdb.set_trace()
        return Mat

    def modify_gt(self, Mat1, Mat2):
        """
            Correctness checked: yes
            Discription: delete extra zeros and produce the groundtruth format we need
            Args:
                Mat1 (numpy): (id), having extra zeros
                Mat2 (numpy): (id), having extra zeros
            Return:
                res (numpy): (num_pairs) 
        """
        for idx, id in enumerate(Mat1):
            if id == 0:
                Mat1 = Mat1[0: idx]
                break
        for idx, id in enumerate(Mat2):
            if id == 0:
                Mat2 = Mat2[0: idx]
                break
        len1 = len(Mat1)
        len2 = len(Mat2)
        num_pairs = len1 * len2
        res = np.zeros(num_pairs)
        for idx1, id1 in enumerate(Mat1):
            for idx2, id2 in enumerate(Mat2):
                if id1 == id2:
                    res[idx1 * len2 + idx2] = 1
        return res

    def wrap_del_tensor_0(self, Mat):
        '''
            Correctness checked: yes
            Discription: numpy -> tensor -> transformation -> numpy
        '''
        ts = self.del_tensor_0(torch.from_numpy(Mat))
        return ts.numpy()

    def samples_balance(self, data, gt):
        """
            Correctness checked: yes
            Discription:
            Args:
                data (numpy): (num_pairs, n, 6)
                gt (numpy): (num_pairs)
            Return:
                data_mod(numpy): (num_pairs_mod, n, 6)
                gt_mod(numpy): (num_pairs_mod)
        """
        dimen_2 = data.shape[1]
        num = np.sum(gt)
        num_t = 0
        ix = gt == 1
        index = []
        data_mod = np.zeros((1, dimen_2, 6))
        gt_mod = np.zeros((1))
        for i in range(len(ix)):
            if  ix[i]:
                index.append(i)
                data_i = np.expand_dims(data[i],0)
                gt_i = np.expand_dims(gt[i],0)
                data_mod = np.concatenate((data_mod, data_i))
                gt_mod = np.concatenate((gt_mod, gt_i))
            if not ix[i]:
                if num_t < num+4:
                    index.append(i)
                    data_i = np.expand_dims(data[i],0)
                    gt_i = np.expand_dims(gt[i],0)
                    data_mod = np.concatenate((data_mod, data_i))
                    gt_mod = np.concatenate((gt_mod, gt_i))
                    num_t = num_t + 1

        return data_mod[1:,:,:], gt_mod[1:]