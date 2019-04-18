from model import conv_3d_id_model
from utils import *

import keras.backend as K
from keras import optimizers

data_path = '../matrices/'
save_path = '../checkpoints/visual_model/'
seq_len = 10 # 16 # 10
seq_len = 1
img_x = 48
img_y = 48
ch_n = 1

#batch_size = 128
batch_size = 64
epochs = 100

##### Creating model

K.clear_session()

m = conv_3d_id_model(seq_len, img_x, img_y, ch_n).create()
opti = optimizers.Adam(lr=0.0001)
#m.load_weights('../checkpoints/visual_model/conv_3D_fullbody_face.02-0.94.hdf5')

m.compile(loss='mean_squared_error', optimizer=opti)
#m.compile(loss=ccc_error, optimizer=opti)
#print('loaded latest model with best weights')

##### load files and make generator

lbl_tr = np.load(data_path + "face_lbl_tr.npy", mmap_mode='r').reshape(-1,1)
#lbl_tr = np.load(data_path + "lbl_tr_small.npy", mmap_mode='r').reshape(-1,1)
print("train labels loaded with shape:", lbl_tr.shape)

video_train_x = np.load(data_path + "face_img_tr.npy", mmap_mode='r')
#video_train_x = np.load(data_path + "img_tr_small.npy", mmap_mode='r')
print("train image loaded with shape: ", video_train_x.shape)

lw_gen_tr = light_generator(video_train_x[:],lbl_tr[:],seq_len,batch_size)
steps_per_epoch_tr = lw_gen_tr.stp_per_epoch

lbl_vl = np.load(data_path + "face_lbl_vl.npy", mmap_mode='r').reshape(-1,1)
#lbl_vl = np.load(data_path + "lbl_vl_small.npy", mmap_mode='r').reshape(-1,1)
print("val labels loaded with shape:", lbl_vl.shape)

video_valid_x = np.load(data_path + "face_img_vl.npy", mmap_mode='r')
#video_valid_x = np.load(data_path + "img_vl_small.npy", mmap_mode='r')
print("val image loaded with shape:", video_valid_x.shape)

lw_gen_vl = light_generator(video_valid_x,lbl_vl,seq_len,batch_size)
steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

##### Setting callbacks

save_name = save_path+'face_high_patience_mse.{epoch:02d}-{val_loss:.2f}.hdf5'

bckup_callback = cb.ModelCheckpoint(save_name, 
                                    monitor='val_loss', 
                                    verbose=0, 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='auto', 
                                    period=1)

stop_callback = cb.EarlyStopping(monitor='val_loss', patience=15)

callbacks_list = [
    bckup_callback,
    stop_callback,
    cb.TensorBoard(log_dir="logs/" + day_time)
]

m.fit_generator(   lw_gen_tr.generate(),
                   steps_per_epoch=steps_per_epoch_tr, 
                   epochs=epochs,
                   callbacks=callbacks_list,
                   validation_data = lw_gen_vl.generate(),
                   validation_steps = steps_per_epoch_vl,
                   shuffle=True)

