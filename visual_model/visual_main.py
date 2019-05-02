from model import alphacity_conv3d, alphacity_conv3d_reduced, cnn_lstm, alphacity_timedistributed, alphacity_timedistributed_reduced, tiny_dark_net, cnn_facial_emotion, cnn_GRU
from utils import *

import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

data_path = '../matrices/'

seq_len = 10
img_x = 48
img_y = 48
ch_n = 1

batch_size = 64
epochs = 100

##### Creating model

K.clear_session()

# change this to whichever model you are testing
m = cnn_facial_emotion(seq_len, img_x, img_y, ch_n).create()

opti = optimizers.Adam(lr=0.0001)
m.compile(loss=ccc_error, optimizer=opti)

##### load files and make generator

lbl_tr = np.load(data_path + "face_lbl_tr.npy", mmap_mode='r')
print("train labels loaded with shape:", lbl_tr.shape)

video_train_x = np.load(data_path + "face_img_tr.npy", mmap_mode='r')
print("train image loaded with shape: ", video_train_x.shape)

gen_tr = video_generator(video_train_x[:],lbl_tr[:],seq_len,batch_size)
steps_per_epoch_tr = gen_tr.stp_per_epoch

lbl_vl = np.load(data_path + "face_lbl_vl.npy", mmap_mode='r')
print("val labels loaded with shape:", lbl_vl.shape)

video_valid_x = np.load(data_path + "face_img_vl.npy", mmap_mode='r')
print("val image loaded with shape:", video_valid_x.shape)

gen_vl = video_generator(video_valid_x,lbl_vl,seq_len,batch_size)
steps_per_epoch_vl = gen_vl.stp_per_epoch

##### Setting callbacks
save_name = '../checkpoints/timedistributed_FINAL.{epoch:01d}-{val_loss:.2f}.hdf5'

bckup_callback = cb.ModelCheckpoint(save_name, 
                                    monitor='val_loss', 
                                    verbose=0, 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='auto', 
                                    period=1)

stop_callback = cb.EarlyStopping(monitor='val_loss', patience=10)

callbacks_list = [
    bckup_callback,
    stop_callback,
    cb.TensorBoard(log_dir="logs/" + day_time)
]

m.fit_generator(  gen_tr.generate(),
                  steps_per_epoch=steps_per_epoch_tr, 
                  epochs=epochs,
                  callbacks=callbacks_list,
                  validation_data = gen_vl.generate(),
                  validation_steps = steps_per_epoch_vl,
                  shuffle=True)

#### testing model
print "testing model..."

lbl_test = np.load(data_path + "face_lbl_test.npy", mmap_mode='r')
print("test labels loaded with shape:", lbl_vl.shape)

video_test_x = np.load(data_path + "face_img_test.npy", mmap_mode='r')
print("test image loaded with shape:", video_valid_x.shape)

gen_vl = video_generator(video_test_x,lbl_test,seq_len,batch_size)
steps_per_epoch_vl = gen_vl.stp_per_epoch

# add saved weights
checkpoint_filename = '../checkpoints/'
m.load_weights(checkpoint_filename)
preds = m.predict_generator(gen_vl.generate_predict(), steps=gen_vl.stp_per_epoch)
print("lbl shape: ", lbl_test.shape)
print("preds shape: ", preds.shape)

# make them the correct size 
diff = lbl_test.shape[0] - preds.shape[0]
add = np.repeat(np.mean(preds), diff)
preds = np.append(preds, add) 

# compute CCC score
ccc_result = ccc(lbl_test.flatten(), preds.flatten())
print("val_ccc (pearson):{} ({})".format(ccc_result[0], ccc_result[1]))
print("*"*50)

# compute rescaled CCC score
preds_norm = norm_pred(lbl_tr, preds)
ccc_result_norm = ccc(lbl_test.flatten(), preds_norm.flatten())
print("val_ccc_tricks (pearson):{} ({})".format(ccc_result_norm[0], ccc_result_norm[1]))