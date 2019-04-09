from PIL import Image 
from os import listdir, makedirs
from itertools import tee, izip
from skimage import io

import math
import random
import numpy as np

'''
script for preprocessing images into numpy arrays
'''

# full body paths
save_path = '/Users/julialu/cpsc490-senior-project/model_fullybody/trained/'
img_path = '/Users/julialu/cpsc490-senior-project/data/Training/FullBody/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '/Users/julialu/cpsc490-senior-project/data/Training/Annotations/Subject_{0}_Story_{1}.csv'

seq_len = 16
seq_overlap = 1
img_x = 107
img_y = 107 # try to make this 48 x 48
ch_n = 1

def create_label_vector(subject_nums, story_nums):
	lbls = []
	for str_n in subject_nums:
		for sbj_n in story_nums:
			# skip the first 15 values of each story because of the sliding window  
			values = np.loadtxt(lbl_path.format(sbj_n,str_n), skiprows = seq_len) # not -1 because first row has label 'valence'
			lbls.append(values)
	return np.array(lbls).reshape(-1,1)

def window(iterable, size):
	''' https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator '''
	iters = tee(iterable, size)
	for i in xrange(1, size):
		for each in iters[i:]:
			next(each, None)
	return izip(*iters)

def create_img_vec(sbj_n,str_n, seq_len):
	'''
	Given a directory of images from a specific story, convert images into 
	numpy array
	'''
	path = img_path.format(sbj_n,str_n)
	frames_n = listdir(path)
	# sorted by time
	sorted_frames_n = list(np.array(frames_n)[np.argsort([int(x[:-4]) for x in frames_n])])
	img_s = []

	for f_n in sorted_frames_n:
		iii = io.imread(path+f_n)
		print("iii shape", iii.shape)
		iii = (iii-np.mean(iii))/np.std(iii)
		img_s.append(iii[:,:,np.newaxis])

	window_frames = window(img_s, seq_len)
	frames = list(window_frames)
	print("converted ")
	return np.array(frames)

def create_img_dataset(n,img_x,img_y,str_n_s,sbj_n_s):

	img_mat = np.zeros([n, seq_len, 107,107,1]) # original size of image
	# total number of data points given the sliding window 
	idx_srt = 0

	# loops over all pngs
	for str_n in str_n_s:
		for sbj_n in sbj_n_s:
			img_s = create_img_vec(sbj_n,str_n, seq_len)
			print("window frames with shape:", img_s.shape)
			idx_end = idx_srt+img_s.shape[0]
			print("mat")
			img_mat[idx_srt:idx_end,:,:,:] = img_s
			print("adding img_s")
			idx_srt = idx_end
		print("Story number and subject number:", str_n,sbj_n)

	return img_mat


if __name__ == "__main__":

	##### load training data 

	# Story numbers that we are training with
	sbj_n_s = range(1,2)
	#sbj_n_s = range(1,11)
	str_n_s = [1]
	# str_n_s = [1,4,5,8]

	lbl_tr = create_label_vector(sbj_n_s, str_n_s)
	print('train labels loaded with shape: ',lbl_tr.shape) 

	img_tr = create_img_dataset(lbl_tr.shape[0],img_x,img_y,str_n_s,sbj_n_s)
	print('train images loaded with shape: ',img_tr.shape)

	# save to npz file 

	##### load validation data 

	#sbj_n_s = range(1,11)
	sbj_n_s = range(1,2)
	str_n_s = [2] 

	lbl_val = create_label_vector(sbj_n_s, str_n_s)
	img_val = create_img_dataset(img_path, lbl_val.shape[0],img_x,img_y,str_n_s,sbj_n_s)
	# saze to npz file