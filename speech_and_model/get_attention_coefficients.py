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
from tensorflow.keras.models import Model
import tensorflow as tf

TEST = 'test'
VAL = 'val'

MODE = VAL

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

# VIDEO YOU WANT TO EVALUATE
target_sbj = 10
target_story = 1
# FRAME OF VIDEO TO START AT
video_offset = 9450		
MODEL = '../models/multimodal.hdf5'

video_start = 0
found = False
for subject in sbj_n:
	if found:
		break
	for story in str_n:
		if subject == target_sbj and story == target_story:
			found = True
			break
		name = name_format.format(subject, story)
		# count number of frames in original 
		annotations = np.loadtxt(annotations_path + name + '.csv', skiprows=1)
		num_frames = len(annotations)
		print name + ': {} frames'.format(num_frames)
		video_start += num_frames

print 'Video Start: {}'.format(video_start)

with tf.device('/cpu:0'):
#load classification model and latent extractor
	valence_model = load_model(MODEL, custom_objects={'CCC':uf.CCC,'batch_CCC':batch_CCC})

	layer_name = 'att_weights'
	attention_model = Model(inputs=valence_model.input, outputs=valence_model.get_layer(layer_name).output)

	start = video_start + video_offset

	audio_predictor = audio_data[start*frames_per_annotation:(start+SEQ_LENGTH)*frames_per_annotation]
	video_predictor = video_data[start:start+SEQ_LENGTH]

	attention_coefficients = attention_model.predict([audio_predictor[np.newaxis], video_predictor[np.newaxis]])
	valences = valence_model.predict([audio_predictor[np.newaxis], video_predictor[np.newaxis]])
print 'Attention coefficients...'
print attention_coefficients
print 'Predicted valences...'
print valences
print 'True valences...'
print labels[start:start+SEQ_LENGTH]
