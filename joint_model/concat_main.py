import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization, Add, RepeatVector, Lambda, Multiply, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, LSTM

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

print speech_train_x.shape
print speech_valid_x.shape
print train_target.shape
print validation_target.shape

## load dataset for video

video_train_x = np.load("../matrices/face_img_tr.npy", mmap_mode='r')
print "train image loaded with shape: " + str(video_train_x.shape)

lbl_tr = np.load("../matrices/face_lbl_tr.npy", mmap_mode='r')
print "train labels loaded with shape:" + str(lbl_tr.shape)

video_valid_x = np.load("../matrices/face_img_vl.npy", mmap_mode='r')
print "val image loaded with shape:" + str(video_valid_x.shape)

lbl_vl = np.load("../matrices/face_lbl_vl.npy", mmap_mode='r')
print "val labels loaded with shape:" + str(lbl_vl.shape)

#hyperparameters
SEQ_LENGTH = 100
batch_size = 8
num_epochs = 200
lstm1_depth = 250
feature_vector_size = 512  
drop_prob = 0.6

# determined in preprocessing, NOT hyperparameter
frames_per_annotation = 4
multi_input_gen_train = uf.multi_input_generator(speech_train_x, video_train_x, train_target, SEQ_LENGTH, batch_size, frames_per_annotation)
multi_input_gen_val = uf.multi_input_generator(speech_valid_x, video_valid_x, validation_target, SEQ_LENGTH, batch_size, frames_per_annotation)

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
save_name = '../checkpoints/joint_model/joint_model5.{epoch:02d}-{val_loss:.2f}.hdf5'
best_model = ModelCheckpoint(save_name, monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=10)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

#model definition
speech_input = Input(shape=(time_dim, features_dim))

gru = Bidirectional(GRU(lstm1_depth, return_sequences=True))(speech_input)
norm = BatchNormalization()(gru)
norm = TimeDistributed(Dropout(drop_prob))(norm)
reshape = Reshape((SEQ_LENGTH, norm.shape[-1] * frames_per_annotation))(norm)
speech_features = TimeDistributed(Dense(feature_vector_size, activation='relu'))(reshape)

## conv3d network for video model 

img_x = 48 
img_y = 48
ch_n = 1

video_input = Input(shape=(SEQ_LENGTH, img_x, img_y, ch_n), name='video_input')

layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(video_input)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)

layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 

layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 

layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
layer = TimeDistributed(BatchNormalization())(layer) 

layer = TimeDistributed(Flatten())(layer)
layer = TimeDistributed(Dense(1024, activation='relu', name='conv_out', kernel_regularizer=regularizers.l2(0.0001)))(layer)

layer = TimeDistributed(BatchNormalization())(layer) 
layer = TimeDistributed(Dropout(0.45))(layer)
layer = TimeDistributed(Dense(1024, activation='relu', name='conv_out', kernel_regularizer=regularizers.l2(0.0001)))(layer)

video_features = TimeDistributed(Dense(feature_vector_size,activation='relu', name='video_features'))(layer)

## feature concatenation 

fused_features = Concatenate()([video_features, speech_features])
ihidden1 = TimeDistributed(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))(fused_features)
layer = TimeDistributed(BatchNormalization())(fused_features)
layer = TimeDistributed(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))(layer)
layer = TimeDistributed(BatchNormalization())(layer)
layer = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))(layer)
flat = Flatten()(layer)
out = Dense(SEQ_LENGTH, activation='linear')(flat)

#model creation
valence_model = Model(inputs=[speech_input, video_input], outputs=out)
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
	verbose=True)

print "Train loss = " + str(min(history.history['loss']))
print "Validation loss = " + str(min(history.history['val_loss']))
