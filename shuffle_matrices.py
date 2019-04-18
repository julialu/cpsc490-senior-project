import numpy as np
import ConfigParser
np.random.seed(523)

print "loading dataset..."
cfg = ConfigParser.ConfigParser()
cfg.read('config/configOMG.ini')

#load parameters from config file

#load datasets
speech_train_x = np.load('matrices/training_2A_S_predictors.npy')
train_target = np.load('matrices/training_2A_S_target.npy')
speech_valid_x = np.load('matrices/validation_2A_S_predictors.npy')
validation_target = np.load('matrices/validation_2A_S_target.npy')

print speech_train_x.shape
print speech_valid_x.shape
print train_target.shape
print validation_target.shape

## load dataset for video

video_train_x = np.load("matrices/fullbody_img_tr.npy")
print "train image loaded with shape: " + str(video_train_x.shape)

lbl_tr = np.load("matrices/fullbody_lbl_tr.npy")
print "train labels loaded with shape:" + str(lbl_tr.shape)

video_valid_x = np.load("matrices/fullbody_img_vl.npy")
print "val image loaded with shape:" + str(video_valid_x.shape)

lbl_vl = np.load("matrices/fullbody_lbl_vl.npy")
print "val labels loaded with shape:" + str(lbl_vl.shape)

#rescale speech datasets to mean 0 and std 1 (validation with respect
#to training mean and std)
tr_mean = np.mean(speech_train_x)
tr_std = np.std(speech_train_x)
v_mean = np.mean(speech_valid_x)
v_std = np.std(speech_valid_x)
speech_train_x = np.subtract(speech_train_x, tr_mean)
speech_train_x = np.divide(speech_train_x, tr_std)
speech_valid_x = np.subtract(speech_valid_x, tr_mean)
speech_valid_x = np.divide(speech_valid_x, tr_std)

# make sure audio and video labels are the same
for i in range(lbl_tr.shape[0]):
	if lbl_tr[i] != train_target[i]:
		print i, 'Audio', train_target[i], 'Video', lbl_tr[i]

for i in range(lbl_vl.shape[0]):
	if lbl_vl[i] != validation_target[i]:
		print 'Audio', validation_target[i], 'Video', lbl_vl[i]

frames_per_annotation = 4

def frameIndicesToSpeechIndices(frameIndices, fpa):
	speechIndices = np.zeros(frameIndices.shape[0] * fpa, dtype=int)
	frameIndices *= fpa
	for i in range(speechIndices.shape[0]):
		if i % fpa == 0:
			speechIndices[i] = frameIndices[i / fpa]
		else:
			speechIndices[i] = speechIndices[i-1] + 1
	return speechIndices

indices = np.arange(video_train_x.shape[0])
np.random.shuffle(indices)

video_train_x = video_train_x[indices]
train_target = train_target[indices]
speech_train_x = speech_train_x[frameIndicesToSpeechIndices(indices, frames_per_annotation)]

print 'Saving shuffled training matrices...'

np.save('matrices/speech_train_predictors.npy', speech_train_x)
np.save('matrices/video_train_predictors.npy', video_train_x)
np.save('matrices/train_target.npy', train_target)

print 'Training matrices successfully saved'

indices = np.arange(video_valid_x.shape[0])
np.random.shuffle(indices)

video_valid_x = video_valid_x[indices]
validation_target = validation_target[indices]
speech_valid_x = speech_valid_x[frameIndicesToSpeechIndices(indices, frames_per_annotation)]

print 'Saving shuffled validation matrices...'

np.save('matrices/speech_valid_predictors.npy', speech_valid_x)
np.save('matrices/video_valid_predictors.npy', video_valid_x)
np.save('matrices/validation_target.npy', validation_target)

print 'Validation matrices successfully saved'
