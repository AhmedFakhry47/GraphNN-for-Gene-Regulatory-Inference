from data_utils import MULTI_G
from utils import Evaluation
import tensorflow as tf

class Shallow_M (tf.keras.Model):
  def __init__(self,n_classes=1,act='sig',training=True):
    super(Shallow_M,self).__init__()
    self.DenseP   = tf.keras.layers.Dense(256,activation=tf.nn.relu)
    self.BatchN_P = tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_P  = tf.keras.layers.Dropout(rate=0.5)    
    
    self.DenseA   = tf.keras.layers.Dense(256,activation=tf.nn.relu)
    self.BatchN_A = tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_A  = tf.keras.layers.Dropout(rate=0.5)    
    
    self.DenseB   = tf.keras.layers.Dense(256,activation=tf.nn.relu)
    self.BatchN_B = tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_B  = tf.keras.layers.Dropout(rate=0.5)    
    
    self.CnnD     = tf.keras.layers.Conv2D(filters=16,kernel_size=2,)
    self.BatchN_D = tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_D  = tf.keras.layers.Dropout(rate=0.5)
    
    self.CnnE     = tf.keras.layers.Conv2D(filters=8,kernel_size=1,)
    self.BatchN_E = tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_E  = tf.keras.layers.Dropout(rate=0.5)
    
    self.Flatten  = tf.keras.layers.Flatten()
    self.DenseD   = tf.keras.layers.Dense(32,activation=tf.nn.relu)
    self.BatchN_DD= tf.keras.layers.BatchNormalization(momentum=0.9)
    self.DropO_DD = tf.keras.layers.Dropout(rate=0.5)

    if (act == 'sig'):
      self.Predict  = tf.keras.layers.Dense(n_classes,activation=tf.nn.sigmoid)
    else:
      self.Predict  = tf.keras.layers.Dense(n_classes,activation=tf.nn.softmax)

    self.training = training

  def call(self,inputs):
    x = tf.math.divide(inputs[:,0,:],inputs[:,1,:])
    
    p = self.DenseP(inputs[:,0,:])
    p = self.BatchN_P(p)
    p = self.DropO_P(p)

    x = self.DenseA(x)
    x = self.BatchN_A(x,)
    x = self.DropO_A(x,training=self.training)
    
    x = self.DenseB(x)
    x = self.BatchN_B(x,)
    x = self.DropO_B(x,training=self.training)

    d = self.CnnD(inputs[:,:,:,np.newaxis])
    d = self.BatchN_D(d)
    d = self.DropO_D(d)
  
    e = self.CnnE(d)
    e = self.BatchN_E(e)
    e = self.DropO_E(e)
    
    d = self.Flatten(e)
    d = self.DenseD(d)
    d = self.BatchN_DD(d)
    d = self.DropO_DD(d)

    out = tf.keras.layers.concatenate([x,p,d])

    return self.Predict(out)


def FindrML_TRAINER(*params,**DATA):
  if DATA:
    DATA_X,DATA_Y = DATA['DATA']['data']
    TRAIN_I,VAL_I,TEST_I = DATA['DATA']['index']

  if params:
    epochs,batchsize,n_classes = params[0],params[1],params[2]
  else:
    epochs,batchsize,n_classes = 300,8,4
  
  TRAIN_G = MULTI_G(DATA_X[TRAIN_I],DATA_Y[TRAIN_I][:,:n_classes],batch_size=batchsize,n_classes=n_classes)
  VAL_G   = MULTI_G(DATA_X[VAL_I]  ,DATA_Y[VAL_I][:,:n_classes]  ,batch_size=1 ,n_classes=n_classes)
  TEST_G  = MULTI_G(DATA_X[TEST_I] ,DATA_Y[TEST_I][:,:n_classes] ,batch_size=1 ,n_classes=n_classes)

  FindrML  = Shallow_M(n_classes=n_classes,act='sig')

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=2500,
    decay_rate=0.8)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  FindrML.compile(optimizer=optimizer,loss='BinaryCrossentropy' ,
                  metrics=['BinaryAccuracy',tf.keras.metrics.AUC(multi_label=True)])

  earlystop = tf.keras.callbacks.EarlyStopping('val_loss',patience=100,restore_best_weights=True)
  evaluator = Evaluation(VAL_G, DATA_Y[VAL_I], TEST_G, DATA_Y[TEST_I],multi=True)

  FindrML.fit(TRAIN_G,epochs=epochs,validation_data=VAL_G,callbacks=[earlystop],verbose=0)
  
  return FindrML
  