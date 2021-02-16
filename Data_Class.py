from sklearn.metrics import roc_curve,roc_auc_score
import tensorflow as tf
import numpy as np
import json

class DATA_Extractor:
  def __init__(self,filedir):
    self.DATA_F = open(filedir,'r')
    self.GeneT  = json.load(self.DATA_F)
    
    self.Data   = {'geneEx':[],'label':[]}

  
  def extract(self,Nof_Networks=10,specify=[]):
    if (len(specify) == 0):
      selector = self.GeneT.keys()
    else:
      selector = specify

    for key in selector:
      for pair in self.GeneT[key]:
        self.Data['geneEx'].append([self.GeneT[key][pair]['Expression_A'],self.GeneT[key][pair]['Expression_B']]) 
        self.Data['label'].append(self.GeneT[key][pair]['Label'])
    
    self.Data['geneEx'] = np.array(self.Data['geneEx'], dtype=np.float64)
    self.Data['label']  = np.array(self.Data['label'])


class MULTI_G(tf.keras.utils.Sequence):
  def __init__(self,genes,labels,batch_size,n_classes=1,shuffle=True):
    self.batch_size = batch_size
    self.labels     = labels
    self.genes      = genes
    self.n_classes  = n_classes
    self.shuffle    = shuffle
    self.dim = self.genes.shape[1:]
    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.genes) / self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    X, y = self.__data_generation(indexes)
    return X, y

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.genes))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    X = np.empty((self.batch_size, *self.dim))
    y = np.empty((self.batch_size,self.n_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      #Melspec
      X[i,]  = self.genes[ID]

      # Store class
      y[i]  = self.labels[ID]

    return X, y