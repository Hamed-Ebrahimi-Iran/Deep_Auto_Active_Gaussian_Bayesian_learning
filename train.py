
import time
import torch
#from classifier import classifier
import joblib
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from skactiveml.pool import UncertaintySampling,ProbabilisticAL
from skactiveml.utils import unlabeled_indices, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from sklearn.metrics import RocCurveDisplay, auc, pair_confusion_matrix, roc_curve, precision_recall_curve,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay
from utils import ConfusionMatrix, feature,plot_roc, plot_train
#from sklearn.decomposition import PCA

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
  # get current date and time

timestr = time.strftime("%Y%m%d-%H%M%S")
path_model = './ckpt/'
path_plots='./plots/'
filemodel=path_model + timestr +'.joblib'
# # save model

def train(train_loader,test_loader,encoders,model,scaler,device):
  with torch.set_grad_enabled(True):
    #model.train()
    results = {}
    n_reps = 10
    n_folds = 5
    gamma=0.025
    use_stratified = True
    pred = torch.zeros(0)
    pred = list(pred.cpu().detach().numpy())
    train_x, y_true=feature(train_loader,encoders,scaler,device)
    #lda = LDA(n_components=1)
    #train_x = lda.fit_transform(train_x, y_true)
    #pca=PCA(n_components=y_true.shape[0])  
    #train_x=pca.fit_transform(train_x)
    n_cycles=int(train_x.shape[0]-(train_x.shape[0]/n_folds))
    kfold_class = StratifiedKFold if use_stratified else KFold
    accuracies = np.full((n_reps, n_folds, n_cycles), np.nan)
    #classifier(model,train_x, y_true,n_folds)
    clf = SklearnClassifier(model, classes=[0,1],random_state=42)
    qs = ProbabilisticAL(random_state=42, metric='rbf',metric_dict={'gamma':'mean'})
    #qs=UncertaintySampling(method='margin_sampling')
    train_start = time.time()
    for i_rep in range(n_reps):
          update_start = time.time()
          kf = kfold_class(n_splits=n_folds, shuffle=True, random_state=42)
          for i_fold, (train_idx, test_idx) in enumerate(kf.split(train_x, y_true)): # type: ignore
                X_test = train_x[test_idx]
                y_test = y_true[test_idx]
                X_train = train_x[train_idx]
                y_train_true = y_true[train_idx]
                y_train = np.full(shape=y_train_true.shape, fill_value=MISSING_LABEL)

                #qs=UncertaintySampling(method='entropy')
                clf.fit(X_train, y_train)
               
                for c in range(n_cycles):
                    query_idx = qs.query(X=X_train, y=y_train, clf=clf)
                    y_train[query_idx] = y_train_true[query_idx]
                    clf.fit(X_train, y_train)
                    accuracies[int(i_rep), i_fold, c] = clf.score(X_test, y_test) 
          end_update = time.time()
          print('++++++++++ End update model',int(end_update-update_start))           
    end_train = time.time()
    print('********* End Train model',int(end_train-train_start))
    results[('GaussianBayesian', 'AutoActivelearningAL')] = accuracies
    jbmodel=joblib.dump(clf, filemodel)                
    
    test_x_, y_test_=feature(test_loader,encoders,scaler,device)
    #test_x_ = lda.fit_transform(test_x_, y_test_)
    #test_x_=pca.fit_transform(test_x_)
    start_Pridict = time.time()  
    pred_pro = clf.predict_proba(test_x_)
    pred=clf.predict(test_x_)
    End_Pridict = time.time()
    print('########### End predicts model',int(End_Pridict-start_Pridict))
    
    # keep probabilities for the positive outcome only
    pred_pro = pred_pro[:, 1]
    fpr, tpr, threshold = roc_curve(y_test_,pred)
    fpr_pro, tpr_pro, threshold = roc_curve(y_test_,pred_pro)  
    # plot the roc curve
    
    print('fpr, tpr, threshold',fpr, tpr, threshold)
    rec_auc = auc(fpr, tpr) 
    print('rec_auc',rec_auc) 
    precision, recall, th = precision_recall_curve(y_test_, pred)
    print('precision, recall, th',precision, recall, th)
    pr_auc = auc(recall, precision)
    print('pr_auc',pr_auc)

    np.save(path_plots +'pro_fpr.npy', fpr)
    np.save(path_plots +'pro_tpr.npy', tpr)
    np.save(path_plots +'pro_precision.npy', precision)
    np.save(path_plots +'pro_recall.npy', recall)

    plot_roc(fpr, tpr,path_plots)
    plot_roc(fpr_pro, tpr_pro,path_plots)
    ConfusionMatrix(y_true,pred,path_plots)
    plot_train(results,n_cycles,path_plots)

    return jbmodel 
    

#metric_kernels=(['additive_chi2', 'chi2', 'cosine', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid'])    