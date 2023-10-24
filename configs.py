import argparse
from pickle import TRUE

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    parser.add_argument('--view', default=[0,1,2], type=list,
                        help='name for test')
    parser.add_argument('--scene', default=[0,1,2,3,4,5,6,7,8,9], type=list,
                        help='scene to dataset')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle or not')
    parser.add_argument('--epoch', default=50, type=int,
                        help='training epoch')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--eva', default=1, type=float,
                        help='make evaluation after each # epoch')
    parser.add_argument('--bbox', default=False, type=bool,
                        help='when false input keypoints, when true input bbox')
    parser.add_argument('--hungarian', default=False, type=bool,
                        help='whether to use hungarian algorithm for inference (for testing but not training)')        
    parser.add_argument('--loss_comb', default=0, type=int,
                        help='use which loss combination to train the model. 0 for both loss, 1 for foot_point loss, 2 for cross_entropy loss')
    parser.add_argument('--save', default=0, type=int,
                        help='where to save the file')           
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args