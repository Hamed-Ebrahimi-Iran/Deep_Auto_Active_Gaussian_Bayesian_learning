import math
import os as os
import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
import torch as torch
import random as random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# from skactiveml.utils import unlabeled_indices, MISSING_LABEL
# from skactiveml.visualization import plot_decision_boundary, plot_utilities
# from sklearn.metrics.pairwise import chi2_kernel,polynomial_kernel,rbf_kernel,laplacian_kernel
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split, validation_curve # type: ignore
# from matplotlib import pyplot as plt
from encoder import encoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#from sklearn.model_selection import LearningCurveDisplay,learning_curve,ShuffleSplit
#from utils import feature,plot_train
from train import train


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  


setup_seed(int(2333))  # 1577677170  2333

from classifier import classifier
from dataset import Dataset
#from train import train
from test import test
import option

#from utils import Visualizer

#torch.set_default_tensor_type('torch.FloatTensor')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#viz = Visualizer(env='DeepMIL', use_incoming_socket=False)





if __name__ == '__main__':
    
    args = option.parser.parse_args()
    scaler=StandardScaler()
    encoders=encoder(args.feature_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    train_loader = DataLoader(Dataset(args, test_mode=False,transform=None),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
  
    test_loader = DataLoader(Dataset(args, test_mode=True,transform=None),
                              batch_size=32, shuffle=False,  ####
                              num_workers=args.workers, pin_memory=True)


    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    if not os.path.exists('./plots'):
        os.makedirs('./plots')    
    #auc = test(test_loader, model, args, viz, device)
    #priors=[0.1,0.9],
    gnb=GaussianNB()
    
  
    #train_x, y_true=feature(train_loader,encoders,scaler,device)
    jbmodel=train(train_loader,test_loader,encoders,gnb,scaler,device)
    
    #auc = test(test_loader,encoders,model,scaler,args,device)

    #filename = "random_forest.joblib"

    # # save model
    # joblib.dump(rf, filename)

    # # load model
    # loaded_model = joblib.load(filename)


   

















    
    # if epoch % 1 == 0 and not epoch == 0:
    #     torch.save(clf.state_dict(), './Projects/DeepActiveMIL-Bayessian/ckpt/'+args.model_name+'{}-i3d.pkl'.format(epoch))
    #     #auc = test(test_loader, model, args, viz, device)
    #     #print('Epoch {0}/{1}: auc:{2}\n'.format(epoch, args.max_epoch, auc))
    # torch.save(clf.state_dict(), './ckpt/' + args.model_name + 'final.pkl')


