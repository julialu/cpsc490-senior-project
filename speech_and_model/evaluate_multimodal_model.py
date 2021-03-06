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

	audio_data = np.load("../matrices/validation_2A_S_predictors.npy", mmap_mode='r')
	print("audio data loaded with shape:", audio_data.shape)

	str_n = [1]

	annotations_path = '../dataset/Validation/Annotations/'
	model_output_path = '../model_predictions/Validation/'
elif MODE == TEST:
	labels = np.load("../matrices/test_2A_S_target.npy", mmap_mode='r')
	print("labels loaded with shape:", labels.shape)

	video_data = np.load("../matrices/fullbody_img_test_128.npy", mmap_mode='r')
	print("video data loaded with shape:", video_data.shape)

	audio_data = np.load("../matrices/test_2A_S_predictors.npy", mmap_mode='r')
	print("audio data loaded with shape:", audio_data.shape)

	str_n = [3,6,7]

	annotations_path = '../dataset/Test/Annotations/'
	model_output_path = '../model_predictions/Test/'

sbj_n = range(1,11)

name_format = 'Subject_{0}_Story_{1}'

#load config file
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

#get values from config file

REFERENCE_PREDICTORS_LOAD = cfg.get('model', 'reference_predictors_load')

# for audio data normalization
reference_predictors = np.load(REFERENCE_PREDICTORS_LOAD)
ref_mean = np.mean(reference_predictors)
ref_std = np.std(reference_predictors)

audio_data = np.subtract(audio_data, ref_mean)
audio_data = np.divide(audio_data, ref_std)

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

MODEL = '../models/multimodal.hdf5'
#load classification model and latent extractor

# with tf.device('/cpu:0'):
valence_model = load_model(MODEL, custom_objects={'CCC':uf.CCC,'batch_CCC':batch_CCC})

print 'Using model ' + MODEL

def predict_datapoint(audio, video, target):

	final_pred = []
	#compute prediction until last frame
	start = 0
	end = 0
	audio_predictors = []
	video_predictors = []

	# cut audio and video into slices for prediction
	while start < (len(target)-SEQ_LENGTH):
		end = start + SEQ_LENGTH
		audio_temp = audio[start*frames_per_annotation:end*frames_per_annotation]
		video_temp = video[start:end]
		audio_predictors.append(audio_temp)
		video_predictors.append(video_temp)

		start = end

	audio_predictors = np.array(audio_predictors)
	video_predictors = np.array(video_predictors)
	predictions = valence_model.predict([audio_predictors, video_predictors])
	predictions = predictions.reshape(-1)

	#compute prediction for last frame
	audio_last = audio[-int(SEQ_LENGTH*frames_per_annotation):]
	video_last = video[-SEQ_LENGTH:]

	last_pred = valence_model.predict([audio_last[np.newaxis], video_last[np.newaxis]]).reshape(-1)
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
		audio_slice = audio_data[start*frames_per_annotation:end*frames_per_annotation]
		
		if not np.array_equal(label_slice, annotations):
			print '{} label slice and annotations do not match! Num annotations different: {}'.format(name, (label_slice != annotations).sum())
			# raise Exception('{} label slice and annotations do not match!'.format(name))

		predictions = predict_datapoint(audio_slice, video_slice, label_slice)
		print predictions[:10], predictions[-10:]
		target_mean = np.mean(train_labels)
		target_std = np.std(train_labels)
		final_pred = uf.f_trick(predictions, target_mean, target_std)

		#apply butterworth filter
		b, a = butter(1, 0.004, 'low')
		final_pred = filtfilt(b, a, final_pred)

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
