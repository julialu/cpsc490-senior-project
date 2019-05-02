from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical, Sequence

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as cb

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr

from scipy.signal import decimate

import time
from  math import ceil

day_time = time.strftime("%Y-%m-ad_%H_%M_%S")

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

def norm_pred(lbl,pred):

  s0 = np.std(lbl.flatten())
  V = pred.flatten()
  m1 = np.mean(pred.flatten())
  s1 = np.std(pred.flatten())
  m0 = np.mean(lbl.flatten())

  norm_pred = s0*(V-m1)/s1+m0

  return norm_pred


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

class video_generator():

  def __init__(self,x2,y,seq_len,batch_size):
    
    self.x2 = x2
    self.y = y
    
    self.seq_len = seq_len
    self.sample_size = self.y.shape[0]

    self.h = self.x2.shape[1]
    self.w = self.x2.shape[2]
    self.c = self.x2.shape[3]

    self.idx_s = np.arange(0, self.sample_size-self.seq_len + 1, self.seq_len)
    self.batch_size = batch_size
    self.stp_per_epoch = int(ceil(float(len(self.idx_s))/self.batch_size))

  def generate(self):
     '''generates sequences of data for training, shuffles every step'''
     while True:
      np.random.shuffle(self.idx_s)      

      for b in range(self.stp_per_epoch):


        rnd_idx = self.idx_s[b*self.batch_size:(b+1)*self.batch_size]

        x2b = np.empty([len(rnd_idx),self.seq_len,self.h,self.w,self.c])
        yb = np.empty([len(rnd_idx),self.seq_len])
        
        for i in range(len(rnd_idx)):
          
          ri = rnd_idx[i]
          x2b[i,:,:,:,:] = self.x2[ri:ri+self.seq_len,:,:,:]
          yb[i,:] = self.y[ri:ri+self.seq_len]

        yield x2b, yb

  def generate_predict(self):
    '''generates sequences of data without shuffling the data'''
    while True:
      for b in range(self.stp_per_epoch):

        rnd_idx = self.idx_s[b*self.batch_size:(b+1)*self.batch_size]

        x2b = np.empty([len(rnd_idx),self.seq_len,self.h,self.w,self.c])
        yb = np.empty([len(rnd_idx),self.seq_len])
        
        for i in range(len(rnd_idx)):
          
          ri = rnd_idx[i]
          x2b[i,:,:,:,:] = self.x2[ri:ri+self.seq_len,:,:,:]
          yb[i,:] = self.y[ri:ri+self.seq_len]

        yield x2b, yb