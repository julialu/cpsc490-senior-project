from os import listdir
from PIL import Image 

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import pandas as pd

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

'''
util functions for converting images into numpy arrays
'''

def create_img_vec(img_path,img_x, img_y,sbj_n,str_n):

  path = img_path.format(sbj_n,str_n)
  frames_n = listdir(path)
  sorted_frames_n = list(np.array(frames_n)[np.argsort([int(x[:-4]) for x in frames_n])])
  img_s = []

  for f_n in sorted_frames_n:
      # iii = Image.open(path+f_n)
      iii = io.imread(path+f_n)
      # iii = iii.resize((img_x, img_y), Image.ANTIALIAS)
      iii = iii / 255.
      img_s.append(iii[:,:,np.newaxis])
  
  return np.array(img_s)

def create_img_dataset(img_path,n,img_x,img_y,ch_n,str_n_s,sbj_n_s):

    img_mat = np.zeros([n,img_x,img_y,1])
    idx_srt = 0

    # loops over all pngs
    for sbj_n in sbj_n_s:

        for str_n in str_n_s:

            img_s = create_img_vec(img_path,img_x,img_y,sbj_n,str_n)
            idx_end = idx_srt+img_s.shape[0]
            img_mat[idx_srt:idx_end,:,:,:] = img_s
            idx_srt = idx_end
            print("Story number and subject number:", str_n,sbj_n)
            
    return img_mat

def make_id_vector(str_n_s,sbj_n_s,lbl_path):
    
    id_s = []

    for sbj_n in sbj_n_s:

        for str_n in str_n_s:
            print("Subject number and Story number:", sbj_n,str_n)
            lbl = np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1)

            id = np.zeros([len(lbl),len(sbj_n_s)])
            id[:,sbj_n-1]=1

            id_s.append(id)
            
    return np.array(id_s)