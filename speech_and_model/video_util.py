# from os import listdir, makedirs
# from PIL import Image 
# from keras import losses
# from keras.utils import to_categorical, Sequence

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as cb

# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image

# import pandas as pd

# from skimage import io
# from skimage.transform import resizes
# from skimage.color import rgb2gray

# from sklearn.preprocessing import MinMaxScaler

# from scipy.stats import pearsonr

# from scipy.signal import decimate, butter, lfilter, freqz

# import matplotlib.pyplot as plt

PATIENCE=3

import time

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")

def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)

    return ccc, rho


def ccc_error(y_true, y_pred):
    true_mean = K.mean(y_true)
    true_variance = K.var(y_true)
    pred_mean = K.mean(y_pred)
    pred_variance = K.var(y_pred)

    x = y_true - true_mean
    y = y_pred - pred_mean
    rho = K.sum(x * y) / K.sqrt(K.sum(x**2) * K.sum(y**2))
    
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)
    return 1-ccc


class Metrics(cb.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))
    
        ccc_result, rho_result =  ccc(y_val, y_predict)
        
        self._data.append({
           'ccc': ccc_result,
           'rho': rho_result
        })
        print("ccc = %f,  pearson=%f" % (ccc_result[0], rho_result[0]) )
        return

    def get_data(self):
        return self._data


def moving_avg(x,win=300):
    x_av=np.zeros(len(x))
    
    for t in range(len(x)):
        x_av[t]=np.mean(x[t:t+win])

    return x_av
  
  
  
def moving_avg_ctr(x,win=300):
    x_av=np.zeros(len(x))
    
    for t in range(int(win/2),int(len(x-win/2))):
        x_av[t]=np.mean(x[t-int(win/2):t+int(win/2)])

    return x_av
  
  
  
def norm_pred(lbl,pred):

  s0 = np.std(lbl.flatten())
  V = pred.flatten()
  m1 = np.mean(pred.flatten())
  s1 = np.std(pred.flatten())
  m0 = np.mean(lbl.flatten())

  norm_pred = s0*(V-m1)/s1+m0
  
  return norm_pred


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def padder(fff,pad_len,pad_side='f'):
  
  if pad_side=='f':
    pad = np.vstack([fff[0,:] for i in range(pad_len)])
    pad_fff = np.vstack([pad,fff])
  
  elif pad_side=='b':
    pad = np.vstack([fff[-1,:] for i in range(pad_len)])
    pad_fff = np.vstack([fff,pad])
   
  return pad_fff



def expand_pred(x,exp_rate=5):
  
  y = np.zeros([x.shape[0]*exp_rate,1])
  ii = 0
  
  for p in x:
    y[ii:ii+exp_rate] = np.ones([exp_rate,1])*p
    ii+=exp_rate
    
  return y


def sequence_reshape(img,lbl,seq_len):

  img_resh = []
  lbl_resh = []

  for i in range(img.shape[0]-seq_len):

    img_resh.append(img[i:i+seq_len,:,:,:])
    lbl_resh.append(lbl[i+seq_len,:])

  lbl_resh = np.array(lbl_resh)
  img_resh = np.array(img_resh)
  
  print('image shape: ',img_resh.shape)
  print('label shape: ',lbl_resh.shape)
  
  return img_resh, lbl_resh

class light_generator():
   
  def __init__(self,x,y,seq_len,batch_size):
    '''
    x - img_tr[:] 
    y - lbl_tr[:]
    seq_len - seq_len
    batch_size - batch_size
    '''
    self.x = x
    self.y = y
    
    self.seq_len = seq_len
    self.sample_size = self.x.shape[0]
    
    self.h = self.x.shape[1]
    print('h', self.h)
    self.w = self.x.shape[2]
    print('w', self.w)
    self.c = 1 #elf.x.shape[3]
    print('c', self.w)

    self.idx_s = np.arange(self.sample_size-self.seq_len)
    self.batch_size = batch_size
    self.stp_per_epoch = int(self.sample_size/self.batch_size)
    
   
  def generate(self):
    
    while True:
      
      for b in range(self.stp_per_epoch):

        np.random.shuffle(self.idx_s)
        rnd_idx = self.idx_s[:self.batch_size]

        
        xb = np.empty([self.batch_size,self.seq_len,self.h,self.w,self.c])
        yb = np.empty([self.batch_size,1])
        
        for i in range(len(rnd_idx)):
          
          ri = rnd_idx[i]
          xb[i,:,:,:,:] = self.x[ri:ri+self.seq_len,:,:,:]
          yb[i,:] = self.y[ri+self.seq_len,:]

        yield xb, yb