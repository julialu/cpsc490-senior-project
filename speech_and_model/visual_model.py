import numpy as np
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

SPEECH_TRAIN_TARGET = cfg.get('model', 'training_target_load')
VALIDATION_TARGET = cfg.get('model', 'validation_target_load')

train_target = np.load(SPEECH_TRAIN_TARGET, mmap_mode='r')
validation_target = np.load(VALIDATION_TARGET, mmap_mode='r')

## load dataset for video

video_train_x = np.load("../matrices/fullbody_img_tr_128.npy", mmap_mode='r')
print "train image loaded with shape: " + str(video_train_x.shape)

video_valid_x = np.load("../matrices/fullbody_img_vl_128.npy", mmap_mode='r')
print "val image loaded with shape:" + str(video_valid_x.shape)

#hyperparameters
SEQ_LENGTH = 100
batch_size = 32
num_epochs = 200
feature_vector_size = 512 # right now my model assumes that this is 8, might need to change the dense layers if you make it anything higher than 8 

regularization_lambda = 0.0001
reg = regularizers.l2(regularization_lambda)

# determined in preprocessing, NOT hyperparameter
frames_per_annotation = 4
video_gen_train = uf.video_generator(video_train_x, train_target, SEQ_LENGTH, batch_size, frames_per_annotation)
video_gen_val = uf.video_generator(video_valid_x, validation_target, SEQ_LENGTH, batch_size, frames_per_annotation)

# reg = regularizers.l2(regularization_lambda)
sgd = optimizers.SGD(lr=0.001, decay=0.003, momentum=0.5)
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#custom loss
def batch_CCC(y_true, y_pred):
    CCC = uf.CCC(y_true, y_pred)
    CCC = CCC /float(batch_size)
    CCC = 1-CCC
    return CCC


#callbacks
best_model = ModelCheckpoint('../models/video_model_32_64_128_256_rnn512.hdf5', monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=7)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

## conv3d network for video model 

seq_len = 16
img_x = 128
img_y = 128
ch_n = 1

with tf.device('/gpu:1'):
	video_input = Input(shape=(SEQ_LENGTH, img_x, img_y, ch_n), name='video_input')

	layer = TimeDistributed(Conv2D(32, kernel_size=(7, 7), strides=(2,2), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv1')(video_input)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool1')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm1')(layer) 

	# with tf.device('/gpu:1'):
	layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv2')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool2')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm2')(layer) 

	layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv3')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool3')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm3')(layer)

	layer = TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_conv4')(layer)
	layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'), name='video_pool4')(layer)
	layer = TimeDistributed(BatchNormalization(), name='video_norm4')(layer) 

with tf.device('/gpu:2'):
	layer = TimeDistributed(Flatten(), name='video_flat')(layer)
	layer = Bidirectional(GRU(512, return_sequences=True, kernel_regularizer=reg), name='video_gru')(layer)
	layer = BatchNormalization(name='video_norm5')(layer)

	video_features = TimeDistributed(Dense(feature_vector_size,activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='video_features')(layer)

	hidden1 = TimeDistributed(Dense (128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='hidden1')(video_features)
	norm1 = TimeDistributed(BatchNormalization(), name='hidden_norm1')(hidden1)
	hidden2 = TimeDistributed(Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='hidden2')(norm1)
	norm2 = TimeDistributed(BatchNormalization(), name='hidden_norm2')(hidden2)
	hidden3 = TimeDistributed(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=reg), name='hidden3')(norm2)
	norm3 = TimeDistributed(BatchNormalization(), name='hidden_norm3')(hidden3)
	norm5 = BatchNormalization(name='final_norm')(finalgru)
	flat = Flatten(name='flat_out')(norm5)
	out = Dense(SEQ_LENGTH, activation='linear', kernel_regularizer=reg, name='out')(flat)

	#model creation
	valence_model = Model(inputs=[video_input], outputs=out)
	valence_model.compile(loss='mse', optimizer=opt)

	print valence_model.summary()

	#model training
	print 'Training...'
	history = valence_model.fit_generator(
		video_gen_train.generate(), 
		steps_per_epoch=video_gen_train.stp_per_epoch,
		epochs = num_epochs, 
		validation_data=video_gen_val.generate(),
		validation_steps=video_gen_val.stp_per_epoch,
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
