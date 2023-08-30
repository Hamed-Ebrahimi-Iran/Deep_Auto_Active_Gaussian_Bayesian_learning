import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init 




#torch.set_default_tensor_type('torch.FloatTensor')
#torch.set_printoptions(profile="full")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class encoder(nn.Module):
    def __init__(self,n_features):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        
        self.dropout = nn.Dropout(0.6)
        self.BatchNorm1d=nn.BatchNorm1d
        self.BatchNorm2d=nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.LeakyReLU=nn.LeakyReLU()
        self.GELU=nn.GELU()
        #selef.GaussianNLLLoss=nn.GaussianNLLLoss()
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init)

    def forward(self,inputs):
        
        
       

        features = self.LeakyReLU(self.fc1(inputs))
        features = self.dropout(features)
        #features=self.BatchNorm1d(512)

        features = self.LeakyReLU(self.fc2(features))
        features = self.dropout(features)
        #features = self.BatchNorm1d(128)

        # features = self.LeakyReLU(self.fc3(features))
        # features = self.dropout(features)
        # #features= self.BatchNorm1d(32)


        # features = self.LeakyReLU(self.fc4(features))
        # features = self.dropout(features)
        # #features=self.BatchNorm1d(features)

        # features = self.LeakyReLU(self.fc5(features))
        # features = self.dropout(features)
       

        #features = self.sigmoid(features) 

        return features