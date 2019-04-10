#CONVOLUTIONAL NEURAL NETWORK
#tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization, Add, RepeatVector, Lambda, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, Dense, Dropout, Flatten, LSTM, Reshape, TimeDistributed, Input, concatenate, BatchNormalization

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
# tr_mean = np.mean(speech_train_x)
# tr_std = np.std(speech_train_x)
# v_mean = np.mean(speech_valid_x)
# v_std = np.std(speech_valid_x)
# speech_train_x = np.subtract(speech_train_x, tr_mean)
# speech_train_x = np.divide(speech_train_x, tr_std)
# speech_valid_x = np.subtract(speech_valid_x, tr_mean)
# speech_valid_x = np.divide(speech_valid_x, tr_std)

# #normalize target between 0 and 1
# train_target = np.multiply(train_target, 0.5)
# train_target = np.add(train_target, 0.5)
# validation_target = np.multiply(validation_target, 0.5)
# validation_target = np.add(validation_target, 0.5)

print speech_train_x.shape
print speech_valid_x.shape
print train_target.shape
print validation_target.shape


#hyperparameters
batch_size = 100
num_epochs = 200
lstm1_depth = 250
feature_vector_size = 256
drop_prob = 0.3
dense_size = 100
regularization_lambda = 0.01

# determined in preprocessing, NOT hyperparameter
frames_per_annotation = 4
video_train_x = np.zeros((train_target.shape[0], 128, 128, 1))
multi_input_gen_train = uf.multi_input_generator(speech_train_x, video_train_x, train_target, SEQ_LENGTH, batch_size, frames_per_annotation)
video_valid_x = np.zeros((validation_target.shape[0], 128, 128, 1))
multi_input_gen_val = uf.multi_input_generator(speech_valid_x, video_valid_x, validation_target, SEQ_LENGTH, batch_size, frames_per_annotation)

# count = 0
# for [x1b, x2b], y in multi_input_gen_train.generate():
# 	print x1b.shape, x2b.shape, y.shape

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
best_model = ModelCheckpoint(NEW_MODEL, monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=5)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

#model definition
speech_input = Input(shape=(time_dim, features_dim))

gru = Bidirectional(GRU(lstm1_depth, return_sequences=False))(speech_input)
norm = BatchNormalization()(gru)
speech_features = Dense(feature_vector_size, activation='linear')(norm)

## load video data and make generators

img_tr = pickle.load(open("fullbody_img_tr.pkl"))
print ("train image loaded with shape:", img_tr.shape)

lbl_tr = pickle.load(open("fullbody_lbl_tr.pkl"))
print ("train labels loaded with shape:", lbl_tr.shape)

lw_gen_tr = light_generator(img_tr[:],lbl_tr[:],seq_len,batch_size)

img_vl = pickle.load(open("fullbody_img_vl.pkl"))
print ("val image loaded with shape:", img_vl.shape)

lbl_vl = pickle.load(open("fullbody_img_tr.pkl"))
print ("val labels loaded with shape:", lbl_vl.shape)

lw_gen_vl = light_generator(img_vl,lbl_vl,seq_len,batch_size)

## conv3d network for video model 

seq_len = 16
img_x = 48 
img_y = 48
ch_n = 1

batch_size = 128 # (use for the generator)

video_input = Input(shape=(SEQ_LENGTH, img_x, img_y, ch_n), name='video_input')

layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(video_input)
layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
layer = BatchNormalization()(layer) 

layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
layer = BatchNormalization()(layer) 

layer = Flatten()(layer)
video_features = Dense(features_vector_size,activation='relu', name='video_features')(layer)

# attention weights

att_input = Flatten()(speech_input)
att_dense1 = Dense(64, activation='linear')(att_input)
att_dense2 = Dense(32, activation='linear')(att_dense1)
att_weights = Dense(2, activation='softmax')(att_dense2)

# this results in a [batch_size, feature_vector_size, # inputs] tensor
repeat_att_weights = RepeatVector(feature_vector_size)(att_weights)

w_1 = Lambda(lambda x: x[:,:,0])(repeat_att_weights)
w_2 = Lambda(lambda x: x[:,:,1])(repeat_att_weights)

speech_scaled = Multiply()([w_1, speech_features])
video_scaled = Multiply()([w_2, video_features])

fused_features = Add()([speech_scaled, video_scaled])

drop = Dropout(drop_prob)(fused_features)
hidden1 = Dense(128, activation='linear')(drop)
hidden2 = Dense(64, activation='linear')(hidden1)
out = Dense(1, activation='linear')(hidden2)

#model creation
valence_model = Model(inputs=[speech_input, video_input], outputs=out)
#valence_model.compile(loss=batch_CCC, optimizer=opt)
valence_model.compile(loss=batch_CCC, optimizer=opt)

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
	verbose=True,
	shuffle=True)

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
