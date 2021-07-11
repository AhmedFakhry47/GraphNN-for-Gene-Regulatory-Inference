from matplotlib import pyplot as plt
from data_utils import *
from utils import *
import numpy as np
import os

def start_training(params):
  '''
  This function takes:
  1- Training parameters such as batch size, # of epochs, etc.
  2- Type of metrics to evaluate model performance with
  3- The chosen dataset to train model over

  It initiates three pure functions, one for downloading and balancing the data,
  the second one calls the model and starts training, while the third one saves the model 
  and evaluates other statistical metrics.
  '''
  #Decode parameters
  if params:
    dataset = params['dataset']
    scaler  = params['scale']
    logpars = params['log_p'][:4]
    savedir = params['log_p'][-1]

    epochs,batchsize,n_classes  = params['train_p']['epochs'],params['train_p']['batchsize'],params['train_p']['n_classes']

  if not os.path.isdir(savedir):
    os.mkdir(savedir)
    
  #First, load and balance dataset to make it ready for training 
  DATA_X,DATA_Y,TRAIN_I,VAL_I,TEST_I = load_data(dataset = dataset,scaler=scaler)
  data_stats(DATA_X,DATA_Y,TEST_I,names=['Connection','No-Connection','GeneA->R->GeneB','GeneB->R->GeneA'],savedir=savedir)

  FindrML = FindrML_TRAINER(epochs,batchsize,n_classes,DATA={'data':[DATA_X,DATA_Y],'index':[TRAIN_I,VAL_I,TEST_I]})

  print(FindrML.summary())

  [GetROCcurve,GetPRcurve,GetMCC,SaveModel,SaveDir]

  print('\n- - - - - - - - - - - - - - - - - - - - - -\n')
  print('Metrics Scores:')
  print('- - - - - - - - - - - - - - - - - - - - - -\n')
  if logpars[0]: calc_AUC(True,savedir,'FindrML_ROC',preds=FindrML.predict(DATA_X[TEST_I]),labels=DATA_Y[TEST_I][:,:n_classes],n_classes=n_classes)
  print('-----\n')
  if logpars[1]: calc_PR(preds=FindrML.predict(DATA_X[TEST_I]),labels=DATA_Y[TEST_I][:,:n_classes],n_classes=n_classes,savedir=savedir)
  print('-----\n')
  if logpars[2]: calc_MCC(preds=FindrML.predict(DATA_X[TEST_I]),labels=DATA_Y[TEST_I][:,:n_classes],n_classes=n_classes)
  print('-----\n')
  if logpars[3]: save_model(kerasmodel=FindrML,savedir=savedir,name='FindrML')
  print('- - - - - - - - - - - - - - - - - - - - - -\n')


def start_sweeping(params,sweep_points=10):
  print('\n- - - - - - - - - - - - - - - - - - - - - -\n')
  print('Started Sweeping \n')

  #Decode parameters
  if params:
    dataset = params['dataset']
    logpars = params['log_p'][:4]

    epochs,batchsize,n_classes  = params['train_p']['epochs'],params['train_p']['batchsize'],params['train_p']['n_classes']

  if not os.path.isdir('/content/Data-Performance-Relationship'):
    os.mkdir('/content/Data-Performance-Relationship')
  
  elif not os.path.isdir('/content/Data-Performance-Relationship/DataStats'):
    os.mkdir('/content/Data-Performance-Relationship/DataStats')

  elif not os.path.isdir('/content/Data-Performance-Relationship/RocCurves'):
    os.mkdir('/content/Data-Performance-Relationship/RocCurves')


  DATA = {}

  #Run scripts to load data from cloned repo 
  get_data(dataset)

  #Preprocess Data
  if 'yeastract' in dataset.lower():
    DATA   = preprocess_yeastract('/content')
  else:
    DATA = preprocess_dream('/content')

  #Extract inputX and labelY from the json file
  DATA   = extract(DATA,specify=False)
  
  sweep_arr = np.arange(0.001,sweep_points*0.001,0.001)
  
  micros    = []
  prs       = []
  mccs      = []
  dpoint    = []

  for i,scale in enumerate(list(sweep_arr)):

    if (i%5 == 0): print('Test Case #{} has started'.format(i))

    for i in range(5):
      micros_n    = 0
      prs_n       = 0
      mccs_n      = 0

      TRAIN_I,VAL_I,TEST_I = balance(dataset,DATA['label'],tobalance_class=0,scaler=scale)

      data_stats(DATA['geneEx'],DATA['label'],TEST_I,names=['Connection','No-Connection','GeneA->R->GeneB','GeneB->R->GeneA'],savedir='/content/Data-Performance-Relationship/DataStats',name='test_case'+str(scale)+'.png')

      FindrML = FindrML_TRAINER(epochs,batchsize,n_classes,DATA={'data':[DATA['geneEx'],DATA['label']],'index':[TRAIN_I,VAL_I,TEST_I]})

      micros_n += calc_AUC(True,'/content/Data-Performance-Relationship/RocCurves','ROC'+str(scale),preds=FindrML.predict(DATA['geneEx'][TEST_I]),labels=DATA['label'][TEST_I][:,:n_classes],n_classes=n_classes,show=False)
      prs_n    += calc_PR(preds=FindrML.predict(DATA['geneEx'][TEST_I]),labels=DATA['label'][TEST_I][:,:n_classes],n_classes=n_classes,show=False)
      mccs_n   += calc_MCC(preds=FindrML.predict(DATA['geneEx'][TEST_I]),labels=DATA['label'][TEST_I][:,:n_classes],n_classes=n_classes,show=False)[0]


    micros.append(micros_n/5) 
    mccs.append(mccs_n/5)       
    prs.append(prs_n/5)    
    dpoint.append(len(TEST_I))

    if (i%5 == 0): print('\n- - - - - - - - - - - - - - - - - - - - - -\n')

  results_df = pd.DataFrame({'Test Case':dpoint,'Micro-avg AUC':micros,'Average Precision Recall':prs,'Matthews Correlation Coefficients ':[mcc[0] for mcc in mccs]})
  results_df.to_csv('results.csv',index=False)

  plt.figure(figsize=[10, 10])

  plt.plot(dpoint, micros, color='blue' , label='performance against data scale')
  plt.xlabel('Data Scale')
  plt.ylabel('AUC')
  plt.title('Model Performance - Data Size')
  plt.legend(loc="lower right")
  plt.savefig(os.path.join('/content','data_performance.png'))
  plt.close()


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

