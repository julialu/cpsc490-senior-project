import numpy as np
import loadconfig
import os
import pandas as pd
import ConfigParser
import essentia.standard as ess
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from scipy.signal import filtfilt, butter
import utilities_func as uf
import utilities_func as uf
from calculateCCC import ccc2
import feat_analysis2 as fa
import tensorflow as tf

TEST = 'test'
VAL = 'val'

MODE = TEST

#### load data
if MODE == VAL:
	labels = np.load("../matrices/validation_2A_S_target.npy", mmap_mode='r')
	print("labels loaded with shape:", labels.shape)

	video_data = np.load("../matrices/fullbody_img_vl_128.npy", mmap_mode='r')
	print("video data loaded with shape:", video_data.shape)

	str_n = [1]

	annotations_path = '../dataset/Validation/Annotations/'
	model_output_path = '../model_predictions/Validation/'
elif MODE == TEST:
	labels = np.load("../matrices/test_2A_S_target.npy", mmap_mode='r')
	print("labels loaded with shape:", labels.shape)

	video_data = np.load("../matrices/fullbody_img_test_128.npy", mmap_mode='r')
	print("video data loaded with shape:", video_data.shape)

	str_n = [3,6,7]

	annotations_path = '../dataset/Test/Annotations/'
	model_output_path = '../model_predictions/Test/'

sbj_n = range(1,11)

name_format = 'Subject_{0}_Story_{1}'

##### evaluate data

# change parameters depending on model

SEQ_LENGTH = 100
batch_size = 32
frames_per_annotation = 4

#custom loss function
def batch_CCC(y_true, y_pred):
	CCC = uf.CCC(y_true, y_pred)
	CCC = CCC /float(batch_size)
	CCC = 1-CCC
	return CCC

MODEL = '../models/video_model_32_64_128_256_rnn512.hdf5'

with tf.device('/gpu:3'):
	#load classification model and latent extractor
	valence_model = load_model(MODEL, custom_objects={'CCC':uf.CCC,'batch_CCC':batch_CCC})

	print 'Using model ' + MODEL
	print valence_model.summary()
	def predict_datapoint(video, target):

		final_pred = []
		#compute prediction until last frame
		start = 0
		end = 0
		video_predictors = []

		# cut audio and video into slices for prediction
		while start < (len(target)-SEQ_LENGTH):
			end = start + SEQ_LENGTH
			video_temp = video[start:end]
			video_predictors.append(video_temp)

			start = end

		video_predictors = np.array(video_predictors)
		predictions = valence_model.predict(video_predictors)
		predictions = predictions.reshape(-1)

		#compute prediction for last frame
		video_last = video[-SEQ_LENGTH:]

		last_pred = valence_model.predict(video_last[np.newaxis]).reshape(-1)
		missing_samples = len(target) - predictions.shape[0]
		last_pred = last_pred[-missing_samples:]
		final_pred = np.concatenate((predictions, last_pred))
		return final_pred


	train_labels = np.load('../matrices/training_2A_S_target.npy')
	start = 0
	end = 0

	cccs = []
	for subject in sbj_n:
		for story in str_n:
			name = name_format.format(subject, story)
			# count number of frames in original 
			annotations = np.loadtxt(annotations_path + name + '.csv', skiprows=1)
			num_frames = len(annotations)
			print name + ': {} frames'.format(num_frames)

			# extract correct number of frames
			end = end + num_frames
			label_slice = labels[start:end]
			video_slice = video_data[start:end]
			
			if not np.array_equal(label_slice, annotations):
				print '{} label slice and annotations do not match! Num annotations different: {}'.format(name, (label_slice != annotations).sum())
				# raise Exception('{} label slice and annotations do not match!'.format(name))

			predictions = predict_datapoint(video_slice, label_slice)

			target_mean = np.mean(train_labels)
			target_std = np.std(train_labels)
			final_pred = uf.f_trick(predictions, target_mean, target_std)

			#apply butterworth filter
			b, a = butter(3, 0.01, 'low')
			final_pred = filtfilt(b, a, final_pred)
			# final_pred = predictions
			print predictions[:10], predictions[-10:]

			# output to csv file
			preds = { 'valence': final_pred }
			df = pd.DataFrame(preds, columns= ['valence'])

			# change this folder for different models
			df.to_csv(model_output_path + name + '.csv', index=None, header=True)

			ccc = ccc2(label_slice, final_pred)
			print '{} ccc {}'.format(name, ccc)
			cccs.append(ccc)
			start = end

print 'Average ccc: {}'.format(np.mean(cccs))
