from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from matplotlib import pyplot as plt
from tensorflow import keras
import os

def save_roc(*scores,name='ROC_CURVE.png',ROCdir='/content',n_classes=3):
  if scores:
    fpr,tpr,roc_auc = scores[0],scores[1],scores[2]
  else:
    print('Please insert scores to plot the roc curve')
    return

  if n_classes >1 and not isinstance(fpr,dict) : 
    print('please enter scores for all classes to continue ..')
    return

  if not os.path.isdir(ROCdir):
    os.mkdir(ROCdir)
  
    # Plot all ROC curves
  plt.figure(figsize=[8, 8])
  lw = 2
  plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
          color='deeppink', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):
      plt.plot(fpr[i], tpr[i], color=color, lw=lw,
              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Model Performance')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join(ROCdir,name+'.png'))
  plt.close()

def calc_AUC(*saveargs,preds,labels,n_classes=4,show=True):
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  if show: print('Micro-Average = {}\n'.format(roc_auc["micro"]))
  if saveargs:
    if show: print('Saving ROC curve ...\n')
    save,path,name = saveargs[0],saveargs[1],saveargs[2]
    if save:
      save_roc(fpr,tpr,roc_auc,name=name,ROCdir=path,n_classes=n_classes)
      if show: print('ROC curve is saved ..\n')
  
  return roc_auc["micro"]

def save_model(kerasmodel,savedir,name,show=True):

  if '.' in name:
    name = ''.join(name.split('.')[0])
  
  if not os.path.isdir(savedir):
    os.mkdir(savedir)

  kerasmodel.save(os.path.join(savedir,name),save_format='tf')
  if show: print('Model is saved ..\n')

def calc_MCC(preds,labels,n_classes,show=True):
  MCCs = []

  if show: print('Matthews Correlation Coefficients:\n')
  for i in range(n_classes):
    if show: print('Class {}: {}\n'.format(i,matthews_corrcoef(labels[:, i], preds[:, i].round())))
    MCCs.append(matthews_corrcoef(labels[:, i], preds[:, i].round()))
  print('\n')
  return MCCs

def calc_PR(preds,labels,n_classes,savedir=None,show=True):
  # For each class
  precision = dict()
  recall    = dict()
  average_precision = dict()
  
  for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(labels[:, i],preds[:, i])
  
  # A "micro-average": quantifying score on all classes jointly
  precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(),preds.ravel())

  average_precision["micro"] = average_precision_score(labels, preds,average="micro")

  plt.figure(figsize=[10, 10])
  lw = 2
  plt.step(recall['micro'], precision['micro'], where='post')

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
  if savedir : plt.savefig(os.path.join(savedir,'PR'+'.png'))
  plt.close()

  if show: print('Precision recall curve is saved ..\n')
  return average_precision["micro"]


class Evaluation(keras.callbacks.Callback):
  def __init__(self, val_data_gen, val_labels, test_data_gen, test_labels,multi=True):
    super(keras.callbacks.Callback, self).__init__()
    self.test_data   = test_data_gen
    self.val_labels  = val_labels
    self.val_data    = val_data_gen
    self.test_labels = test_labels

    if multi == True:
      self.param = 'ovr'
    else:
      self.param = 'raise'

  def on_epoch_end(self, epoch, logs=None):
    y_preds = self.model.predict_generator(self.val_data)
    print(' | val_auc:', roc_auc_score(self.val_labels[:len(y_preds)], y_preds,multi_class=self.param))

    y_preds = self.model.predict_generator(self.test_data)
    print(' | test_auc:', roc_auc_score(self.test_labels[:len(y_preds)], y_preds,multi_class=self.param))