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

#### load data

labels = np.load("../matrices/validation_2A_S_target.npy", mmap_mode='r')
print("labels loaded with shape:", labels.shape)

video_data = np.load("../matrices/fullbody_img_vl.npy", mmap_mode='r')
print("video data loaded with shape:", video_data.shape)

audio_data = np.load("../matrices/validation_2A_S_predictors.npy", mmap_mode='r')
print("audio data loaded with shape:", audio_data.shape)

##### evaluate data

# change parameters depending on model

SEQ_LENGTH = 200
batch_size = 50
frames_per_annotation = 4

#custom loss function
def batch_CCC(y_true, y_pred):
	CCC = uf.CCC(y_true, y_pred)
	CCC = CCC /float(batch_size)
	CCC = 1-CCC
	return CCC

MODEL = '../models/multimodal_model.hdf5'
#load classification model and latent extractor
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

sbj_n = range(1,11)
# FOR VALIDATION
str_n = [1]
# FOR TEST
#str_n = [3,6,7]

name_format = 'Subject_{0}_Story_{1}'
annotations_path = '../dataset/Validation/Annotations/'
model_output_path = '../model_predictions/Validation/'

# annotations_path = '../dataset/Test/Annotations/'
# model_output_path = '../model_predictions/Test/'


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
			raise Exception('{} label slice and annotations do not match!'.format(name))

		predictions = predict_datapoint(audio_slice, video_slice, label_slice)

		target_mean = np.mean(train_labels)
		target_std = np.std(train_labels)
		final_pred = uf.f_trick(predictions, target_mean, target_std)

		#apply butterworth filter
		b, a = butter(3, 0.01, 'low')
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
