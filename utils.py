# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:23:03 2018

@author: JingGroup
"""
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import torch
import visdom
import numpy as np
from matplotlib import pyplot as plt
from skactiveml.visualization import plot_decision_boundary, plot_utilities


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)

# def process_feat(feat, length):
#     new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
#     r = np.linspace(0, len(feat), length+1, dtype=np.int32)
#     for i in range(length):
#         if r[i]!=r[i+1]:
#             new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
#         else:
#             new_feat[i,:] = feat[r[i],:]
#     return new_feat
# def process_feat(feat, length):
#     new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
#     r = np.linspace(0, len(feat), length+1, dtype=np.int32)
#     for i in range(length):
#         if r[i]!=r[i+1]:
#             new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
#         else:
#             new_feat[i,:] = feat[r[i],:]
#     return new_feat

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int32)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat

def cross_val(X_train,y_train,params):
        model = GaussianNB()
        model.set_params(**params)
        cv_results = cross_val_score(model, X_train, y_train,
                                cv = 30, #10 folds
                                scoring = "accuracy",
                                verbose = 2
                                )
        #return the mean of the 10 fold cross validation
        return cv_results.mean()


# from sklearn.preprocessing import LabelBinarizer
# labelbin = LabelBinarizer()
# labelbin.fit_transform(classes)
# labelbin.classes_

def feature(n_loader,encoder,scaler,device):
    inputs, targets  = next(iter(n_loader)) 
    inputs, targets = inputs.to(device), targets.to(device)
    #feature=encoder(inputs)
    feature=inputs
    inputs, targets=structuer_data(feature, targets,device)
    scaler.fit(inputs)
    inputs=scaler.transform(inputs)
    return inputs, targets

def structuer_data(feature,targets,device):
    nonzero=0
    zero=0
    bs,ncrops,t,f=feature.size()
    features=feature.view(bs* ncrops,t * f)#torch.Size([4, 10, 32, 2048]) -->[40,512]
    
    #features=inputs.view(bs * ncrops , t , f)   ##torch.Size([40, 32, 2048]) #genarate
    nonzero=int(torch.count_nonzero(targets))
    zero = int(targets.numel() - nonzero)
    # label_nor=torch.zeros(((zero * ncrops,t )),dtype=torch.int64,device=device)
    # label_abn=torch.ones((nonzero * ncrops,t),dtype=torch.int64,device=device)
    # new shape
    label_nor=torch.zeros((zero * ncrops),dtype=torch.int64,device=device)
    label_abn=torch.ones(nonzero * ncrops,dtype=torch.int64,device=device)
    targets=torch.cat((label_nor,label_abn),dim=0).to(device)
    features_pca=pca(features,latent_dim=targets.shape[0]).to(device=device)
    features=torch.detach(features_pca).numpy()
    targets=torch.detach(targets).numpy()
    return features,targets

def pca(input, latent_dim):
    U, S, V = torch.pca_lowrank(input, q = latent_dim,niter=5)
    return torch.nn.Parameter(torch.matmul(input, V[:,:latent_dim]))

def plot_train(results,n_cycles,path):
    key = ('GaussianBayesian', 'AutoActivelearningAL')
    result = results[key]
    reshaped_result = result.reshape((-1, n_cycles))
    errorbar_mean = np.mean(reshaped_result, axis=0)
    errorbar_std = np.std(reshaped_result, axis=0)
    plt.errorbar(np.arange(n_cycles), errorbar_mean, errorbar_std,fmt='b',ecolor='g',label=f"({np.mean(errorbar_mean):.4f}) {'AutoActivelearningAL'}", alpha=0.5)
    plt.title('Deep Auto Active Gaussian Bayesian learning')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim(0.0,1.1)
    plt.ioff()
    plt.savefig(path +'Deep_Auto_Active_Gaussian_Bayesian_learning.jpg')
    plt.show(block=True)

def plot_roc(fpr, tpr,path):
    plt.plot(fpr, tpr, marker='.', label='ROC Cruve')
    plt.plot([0, 1], [0, 1], '--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(path +'ROC_Cruve.jpg')
    plt.show(block=True)

def ConfusionMatrix(y_true, pred,path):
    target_names = ['normal', 'Anomaly']
    labels_names = [0,1] 
    cm = confusion_matrix(y_true, pred,labels=labels_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
    disp = disp.plot(cmap='bone',values_format='g')
    #ConfusionMatrixDisplay.from_predictions(y_true, pred,labels=[0,1],cmap='orange')
    plt.savefig(path +'ConfusionMatrix.jpg')
    plt.show(block=True)
    