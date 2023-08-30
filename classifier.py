    
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import time 

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold,validation_curve

from skactiveml.pool import UncertaintySampling,ProbabilisticAL
from skactiveml.utils import unlabeled_indices, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from sklearn.metrics import RocCurveDisplay, auc, pair_confusion_matrix, roc_curve, precision_recall_curve,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import torch
from skactiveml.classifier import SklearnClassifier

from utils import plot_roc


  # get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
timestr = time.strftime("%Y%m%d-%H%M%S")
path = './Projects/DeepActiveMIL-Active_NaiveBayes/ckpt/DeepNaiveBayes'
filename=path+timestr+'.joblib'
# # save model

def classifier(model,train_x, y_true,n_folds):
    var_smoothing=np.asanyarray([1e-8])
    #Cs = np.logspace(-2.3, -1.3, 10)
    epoch=np.arange(1,10)
    acc_train=[]
    acc_test=[]
    k_folds = KFold(n_splits = n_folds)
    ss = ShuffleSplit(train_size=0.8, test_size=0.2, n_splits = 5)
    for epo in epoch:
        train_scores, test_scores = validation_curve(model,train_x, y_true,  param_name='var_smoothing',param_range=var_smoothing, cv=10, scoring='accuracy',error_score='raise') # type: ignore
        print('----------- Done-----------')
        print('train_scores',train_scores)
        print('test_scores',test_scores)
        train_mean = np.mean(train_scores, axis=1)
        train_std =  np.std(train_scores, axis=1)
        test_mean =  np.mean(test_scores, axis=1)
        test_std =   np.std(test_scores, axis=1)      
        acc_train.append(train_mean)
        acc_test.append(test_mean)

  
    #joblib.dump(model, filename)
    # # load model
    # loaded_model = joblib.load(filename)
    # Plot mean accuracy scoresfor training and testing scores
    plt.plot(epoch,acc_train,
            label="Training", color='r')
    plt.plot(epoch,acc_test,
            label="Testing", color='b')
    # Creating the plot
    plt.title("Gaussian navie Bayes Classifier")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

#     for i_fold, (train_idx, test_idx) in enumerate(ss.split(train_x, y_true)):
#     #This GaussianNB instance is not fitted yet. 
#        pred_pro = model.fit(train_x[train_idx]).predict_proba(train_x[test_idx])
#     #pred=clf.predict(test_x_)
#     # keep probabilities for the positive outcome only
#     pred_pro = pred_pro[:, 1]
#     fpr, tpr, threshold = roc_curve(y_true,pred_pro)  
#     # plot the roc curve
#     plot_roc(fpr, tpr)
#     print('fpr, tpr, threshold',fpr, tpr, threshold)
#     np.save(path+'mainfpr.npy', fpr)
#     np.save(path+'maintpr.npy', tpr)
#     rec_auc = auc(fpr, tpr) 
#     print('rec_auc',rec_auc) 
#     precision, recall, th = precision_recall_curve(y_true,pred_pro) 
#     print('precision, recall, th',precision, recall, th)
#     pr_auc = auc(recall, precision)
#     print('pr_auc',pr_auc)
#     ConfusionMatrixDisplay.from_predictions(y_true, pred_pro,labels=[0,1])
#     plt.show(block=True)
#     np.save(path+'maiprecision.npy', precision)
#     np.save(path+'mainrecall.npy', recall)
    