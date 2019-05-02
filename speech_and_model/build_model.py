import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization, Add, RepeatVector, Lambda, Multiply, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, LSTM

from video_util import *

import utilities_func as uf
import loadconfig
import ConfigParser
import matplotlib.pyplot as plt
np.random.seed(1)

print "loading dataset..."
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

#load parameters from config file
NEW_MODEL = cfg.get('model', 'save_model')
SPEECH_TRAIN_PRED = cfg.get('model', 'training_predictors_load')
SPEECH_TRAIN_TARGET = cfg.get('model', 'training_target_load')
SPEECH_VALID_PRED = cfg.get('model', 'validation_predictors_load')
VALIDATION_TARGET = cfg.get('model', 'validation_target_load')
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
print "Training predictors: " + SPEECH_TRAIN_PRED
print "Training target: " + SPEECH_TRAIN_TARGET
print "Validation predictors: " + SPEECH_VALID_PRED
print "Validation target: " + VALIDATION_TARGET

#load datasets
speech_train_x = np.load(SPEECH_TRAIN_PRED, mmap_mode='r')
train_target = np.load(SPEECH_TRAIN_TARGET, mmap_mode='r')
speech_valid_x = np.load(SPEECH_VALID_PRED, mmap_mode='r')
validation_target = np.load(VALIDATION_TARGET, mmap_mode='r')

# #rescale datasets to mean 0 and std 1 (validation with respect
# #to training mean and std)
tr_mean = np.mean(speech_train_x)
tr_std = np.std(speech_train_x)
v_mean = np.mean(speech_valid_x)
v_std = np.std(speech_valid_x)
speech_train_x = np.subtract(speech_train_x, tr_mean)
speech_train_x = np.divide(speech_train_x, tr_std)
speech_valid_x = np.subtract(speech_valid_x, tr_mean)
speech_valid_x = np.divide(speech_valid_x, tr_std)

print speech_train_x.shape
print speech_valid_x.shape
print train_target.shape
print validation_target.shape

## load dataset for video

video_train_x = np.load("../matrices/fullbody_img_tr_128.npy", mmap_mode='r')
print "train image loaded with shape: " + str(video_train_x.shape)

lbl_tr = np.load("../matrices/fullbody_lbl_tr_128.npy", mmap_mode='r')
print "train labels loaded with shape:" + str(lbl_tr.shape)

video_valid_x = np.load("../matrices/fullbody_img_vl_128.npy", mmap_mode='r')
print "val image loaded with shape:" + str(video_valid_x.shape)

lbl_vl = np.load("../matrices/fullbody_lbl_vl_128.npy", mmap_mode='r')
print "val labels loaded with shape:" + str(lbl_vl.shape)

#hyperparameters
SEQ_LENGTH = 100
batch_size = 32
num_epochs = 200
feature_vector_size = 512 # right now my model assumes that this is 8, might need to change the dense layers if you make it anything higher than 8 
regularization_lambda = 0.0001

# determined in preprocessing, NOT hyperparameter
frames_per_annotation = 4
multi_input_gen_train = uf.multi_input_generator(speech_train_x, video_train_x, train_target, SEQ_LENGTH, batch_size, frames_per_annotation)
multi_input_gen_val = uf.multi_input_generator(speech_valid_x, video_valid_x, validation_target, SEQ_LENGTH, batch_size, frames_per_annotation)

reg = regularizers.l2(regularization_lambda)
sgd = optimizers.SGD(lr=0.001, decay=0.003, momentum=0.5)
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#custom loss
def batch_CCC(y_true, y_pred):
    CCC = uf.CCC(y_true, y_pred)
    CCC = CCC /float(batch_size)
    CCC = 1-CCC
    return CCC

time_dim = frames_per_annotation*SEQ_LENGTH
features_dim = speech_train_x.shape[1]

#callbacks
best_model = ModelCheckpoint('../models/multimodal.hdf5', monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=10)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

img_x = 128
img_y = 128
ch_n = 1

## conv3d network for video model 
with tf.device('/gpu:0'):
	video_input = Input(shape=(SEQ_LENGTH, img_x, img_y, ch_n), name='video_input')

	layer = TimeDistributed(Conv2D(32, kernel_size=(7, 7), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv1')(video_input)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool1')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm1')(layer) 

	layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv2')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool2')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm2')(layer) 

	layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv3')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool3')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm3')(layer)


# audio model
with tf.device('/gpu:1'):
	speech_input = Input(shape=(time_dim, features_dim))

	gru = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=reg), name='audio_gru1')(speech_input)
	norm = BatchNormalization(name='audio_norm1')(gru)
	dense = TimeDistributed(Dense(256, kernel_regularizer=reg), name='audio_dense1')(norm)
	norm2 = TimeDistributed(BatchNormalization(), name='audio_norm2')(dense)
	# dense2 = TimeDistributed(Dense(256, kernel_regularizer=reg), name='audio_dense2')(norm2)
	# norm3 = TimeDistributed(BatchNormalization(), name='audio_norm3')(dense2)
	reshape = Reshape((SEQ_LENGTH, norm2.shape[-1] * frames_per_annotation), name='audio_reshape1')(norm2)
	# gru2 = Bidirectional(GRU(512, return_sequences=True, kernel_regularizer=reg), name='audio_gru2')(reshape)
	# norm2 = BatchNormalization(name='audio_norm2')(gru2)
	# dense = TimeDistributed(Dense(1024, kernel_regularizer=reg), name='audio_dense1')(reshape)
	# norm2 = TimeDistributed(BatchNormalization(), name='audio_norm2')(dense)
	speech_features = TimeDistributed(Dense(feature_vector_size, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='sum_audio_features')(reshape)

# attention weights
with tf.device('/gpu:2'):
	flat_speech = Reshape((SEQ_LENGTH, features_dim * frames_per_annotation), name='att_reshape_speech')(speech_input)
	flat_video = TimeDistributed(Flatten(), name='att_flat_video')(video_input)
	att_input = Concatenate(name='att_input')([flat_speech, flat_video])
	att_dense1 = TimeDistributed(Dense(1024, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='att_dense1')(att_input)
	att_norm_1 = TimeDistributed(BatchNormalization(), name='att_norm1')(att_dense1)
	# att_gru = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=reg), name='att_dense1')(att_norm_1)
	# att_norm_2 = BatchNormalization()(att_gru)
	att_dense2 = TimeDistributed(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='att_dense2')(att_norm_1)
	att_norm_2 = TimeDistributed(BatchNormalization(), name='att_norm2')(att_dense2)
	att_dense3 = TimeDistributed(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='att_dense3')(att_norm_2)
	att_norm_3 = TimeDistributed(BatchNormalization(), name='att_norm3')(att_dense3)
	att_weights = TimeDistributed(Dense(2, activation='softmax', kernel_regularizer=reg), name='att_weights')(att_norm_3)

	# this results in a [batch_size, seq_length, feature_vector_size, # inputs] tensor
	repeat_att_weights = TimeDistributed(RepeatVector(feature_vector_size), name='repeat_att_weights')(att_weights)

	w_1 = TimeDistributed(Lambda(lambda x: x[:,:,0]), name='w1')(repeat_att_weights)
	w_2 = TimeDistributed(Lambda(lambda x: x[:,:,1]), name='w2')(repeat_att_weights)

with tf.device('/gpu:3'):
	layer = TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv4')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool4')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm4')(layer) 
	layer = TimeDistributed(Flatten(), name='video_flat')(layer)
	layer = Bidirectional(GRU(512, return_sequences=True, kernel_regularizer=reg), name='video_gru')(layer)
	layer = BatchNormalization(name='video_norm5')(layer)

	video_features = TimeDistributed(Dense(feature_vector_size,activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='sum_video_features')(layer)
	speech_scaled = Multiply()([w_1, speech_features])
	video_scaled = Multiply()([w_2, video_features])

	fused_features = Add()([speech_scaled, video_scaled])
	# flat = Flatten()(fused_features)
	hidden1 = TimeDistributed(Dense (128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='sum_hidden1')(fused_features)
	norm1 = TimeDistributed(BatchNormalization(), name='sum_hidden_norm1')(hidden1)
	hidden2 = TimeDistributed(Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='sum_hidden2')(norm1)
	norm2 = TimeDistributed(BatchNormalization(), name='sum_hidden_norm2')(hidden2)
	hidden3 = TimeDistributed(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='sum_hidden3')(norm2)
	norm3 = TimeDistributed(BatchNormalization(), name='sum_hidden_norm3')(hidden3)
	flat = Flatten(name='sum_flat_out')(norm3)
	out = Dense(SEQ_LENGTH, activation='linear', kernel_regularizer=reg, name='sum_out')(flat)

#model creation
valence_model = Model(inputs=[speech_input, video_input], outputs=out)
valence_model.compile(loss='mse', optimizer=opt)
valence_model.load_weights('../models/audio_model_reg_256_dense256_reshape_128_64_32_reg1_seq100.hdf5', by_name=True)
valence_model.load_weights('../models/video_model_32_64_128_256_rnn512_named.hdf5', by_name=True)

print valence_model.summary()

#model training
print 'Training...'
history = valence_model.fit_generator(
	multi_input_gen_train.generate(), 
	steps_per_epoch=multi_input_gen_train.stp_per_epoch,
	epochs = num_epochs, 
	validation_data=multi_input_gen_val.generate(),
	validation_steps=multi_input_gen_val.stp_per_epoch,
	callbacks=callbacks_list,
	verbose=True)

print "Train loss = " + str(min(history.history['loss']))
print "Validation loss = " + str(min(history.history['val_loss']))


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL PERFORMANCE', size = 15)
plt.ylabel('loss', size = 15)
plt.xlabel('Epoch', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(['train', 'validation'], fontsize = 12)

plt.show()
