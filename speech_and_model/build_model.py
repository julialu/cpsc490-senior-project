#CONVOLUTIONAL NEURAL NETWORK
#tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
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
speech_train_x = np.load(SPEECH_TRAIN_PRED)
train_target = np.load(SPEECH_TRAIN_TARGET)
speech_valid_x = np.load(SPEECH_VALID_PRED)
validation_target = np.load(VALIDATION_TARGET)

#rescale datasets to mean 0 and std 1 (validation with respect
#to training mean and std)
tr_mean = np.mean(speech_train_x)
tr_std = np.std(speech_train_x)
v_mean = np.mean(speech_valid_x)
v_std = np.std(speech_valid_x)
speech_train_x = np.subtract(speech_train_x, tr_mean)
speech_train_x = np.divide(speech_train_x, tr_std)
speech_valid_x = np.subtract(speech_valid_x, tr_mean)
speech_valid_x = np.divide(speech_valid_x, tr_std)

#normalize target between 0 and 1
train_target = np.multiply(train_target, 0.5)
train_target = np.add(train_target, 0.5)
validation_target = np.multiply(validation_target, 0.5)
validation_target = np.add(validation_target, 0.5)

#hyperparameters
batch_size = 100
num_epochs = 200
lstm1_depth = 250
feature_vector_size = 512
drop_prob = 0.3
dense_size = 100
regularization_lambda = 0.01

reg = regularizers.l2(regularization_lambda)
sgd = optimizers.SGD(lr=0.001, decay=0.003, momentum=0.5)
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#custom loss
def batch_CCC(y_true, y_pred):
    CCC = uf.CCC(y_true, y_pred)
    CCC = CCC /float(batch_size)
    CCC = 1-CCC
    return CCC

time_dim = speech_train_x.shape[1]
features_dim = speech_train_x.shape[2]

#callbacks
best_model = ModelCheckpoint(NEW_MODEL, monitor='val_loss', save_best_only=True, mode='min')  #save the best model
early_stopping_monitor = EarlyStopping(patience=5)  #stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

#model definition
speech_input = Input(shape=(time_dim, features_dim))
print (time_dim, features_dim)
gru = Bidirectional(GRU(lstm1_depth, return_sequences=True))(speech_input)
norm = BatchNormalization()(gru)
speech_features = TimeDistributed(Dense(feature_vector_size, activation='linear'))(norm)


# TO DO VIDEO MODEL
# video_input = 
# video_features = TimeDistributed(Dense(feature_vector_size, activation='linear'))(norm)

# attention weights
# concatenate inputs
att_dense1 = TimeDistributed(Dense(64, activation='linear'))(speech_input)
att_dense2 = TimeDistributed(Dense(32, activation='linear'))(att_dense1)
att_weights = TimeDistributed(Dense(2, activation='linear'))(att_dense2)

fused_features = TimeDistributed(Add()[speech_features, video_features]) # how to do weighted sum?

drop = Dropout(drop_prob)(fused_features)
flat = Flatten()(drop)
out = Dense(SEQ_LENGTH, activation='linear')(flat)

#model creation
valence_model = Model(inputs=[speech_input, video_input], outputs=out)
#valence_model.compile(loss=batch_CCC, optimizer=opt)
valence_model.compile(loss=batch_CCC, optimizer=opt)

print valence_model.summary()

#model training
history = valence_model.fit([speech_train_x, video_train_x], train_target, epochs = num_epochs, validation_data=([speech_valid_x, video_valid_x],validation_target), callbacks=callbacks_list, batch_size=batch_size, shuffle=True)

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
