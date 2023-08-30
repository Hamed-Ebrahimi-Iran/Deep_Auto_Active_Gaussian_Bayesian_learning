import matplotlib.pyplot as plt
from skactiveml.classifier import SklearnClassifier
import torch
from skactiveml.utils import unlabeled_indices, MISSING_LABEL
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

from utils import feature

def test(test_loader,encoders,model,scaler,args,device):
    with torch.no_grad():
        #model.eval()
        pred = torch.zeros(0)

        test_x, y_test=feature(test_loader,encoders,scaler,device)
        y_train = np.full(shape=y_test.shape, fill_value=MISSING_LABEL)
        for train_x, y_true in enumerate((test_x, y_test)):

            score=model.predict(test_x)
             
               






        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred) 
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)  
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        # plt.plot_lines('pr_auc', pr_auc)
        # plt.plot_lines('auc', rec_auc)
        # plt.lines('scores', pred)
        # plt.lines('roc', tpr, fpr)
        return rec_auc, pr_auc

