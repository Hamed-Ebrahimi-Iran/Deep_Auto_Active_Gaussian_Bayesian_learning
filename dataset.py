import numpy as np
import torch as torch
import torch.utils.data as data
from utils import process_feat
from sklearn.preprocessing import StandardScaler

#torch.set_default_tensor_type('torch.cuda.FloatTensor')



# class PyTMinMaxScalerVectorized(object):
#     """
#     Transforms each channel to the range [0, 1].
#     scikit learn true two dim <3
#     """
#     def __call__(self, tensor):
#         dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
#         dist[dist==0.] = 1.
#         scale = 1.0 /  dist
#         tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
#         return tensor
    

# MinMax = PyTMinMaxScalerVectorized() 
#########################################################################

class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False):
        self.modality = args.modality
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list
        self.num_frame = 0
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.normal_flag='Normal_'

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
              self.list = self.list 

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0
        #labels = self.get_labels(self.list[index].strip('\n'))  # get video level label 0/1
        features = np.array(np.load(self.list[index].strip('\n')),dtype=np.float32)
        #features= np.array(features,dtype=np.float32) # use nomalizer  -->int32 defult
        #print(features)
        if self.tranform is not None:
            feature = self.tranform(features)
        if self.test_mode:
            # name = os.path.basename(self.list[index].strip('\n'))
           # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            test_feature=np.asarray(divided_features, dtype=np.float32)
            divided_features =torch.from_numpy(test_feature) # np-> tensor
            return divided_features, label
        else:
           # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
           
            normal_feature=np.asarray(divided_features, dtype=np.float32)
            #divided_features=MinMax(normal_feature)
            
            divided_features =torch.from_numpy(normal_feature) # np-> tensor
            return divided_features, label
            # features = process_feat(features, 32)
            # return features

    def get_labels(self,name_index):

        if name_index.find('Normal_'):
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
    
    
#zi = 2 * ((xi – xmin) / (xmax – xmin)) – 1.        
##normal_feature=2 * ((normal_feature - np.amin(normal_feature))/ (np.amax(normal_feature)-np.amin(normal_feature))) -1   #[-1,1]     
#normal_feature=(normal_feature - np.amin(normal_feature)) / (np.amax(normal_feature)-np.amin(normal_feature))   #[0,1]   
#scaler.fit(_normal_feature)
#normal_feature=scaler.transform(_normal_feature)
#X_std = (_normal_feature - _normal_feature.min(axis=0)) / (_normal_feature.max(axis=0) - _normal_feature.min(axis=0))
#X_scaled = X_std * (_normal_feature.max(axis=0) - _normal_feature.min(axis=0)) + _normal_feature.min(axis=0)