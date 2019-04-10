# from fullbody_model import conv_3d_id_model
from utils import *

# import keras.backend as K
# from keras import optimizers

import pickle 

# change to the directories that hold the images for the  
# save_path = '/Users/julialu/cpsc490-senior-project/trained/conv_model_fullbody'
img_path = '../dataset/Training/FullBody/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '../dataset/Training/Annotations/Subject_{0}_Story_{1}.csv'

seq_len = 16
seq_len = 1
img_x = 48
img_y = 48
ch_n = 1

# batch_size = 128
# epochs = 100

##### Creating model

# K.clear_session()

# m = conv_3d_id_model(seq_len, img_x, img_y, ch_n, id_len).create()
# opti = optimizers.Adam(lr=0.0001)

# m.compile(loss=ccc_error, optimizer=opti)

##### load training and make generator

# refers to the story and subject #s in video names
sbj_n_s = range(1,11)
str_n_s = [1,4,5,8]

lbl_tr = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
pickle.dump(lbl_tr, open("fullbody_lbl_tr.pkl", "w"))
print('train labels loaded with shape: ',lbl_tr.shape) # (323675, 1)

img_tr = create_img_dataset(img_path, lbl_tr.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
pickle.dump(img_tr, open("fullbody_img_tr.pkl", "w"))
print('train images loaded with shape: ',img_tr.shape) # (323675, 48, 48, 1)

# lw_gen_tr = light_generator(img_tr[:],lbl_tr[:],seq_len,batch_size)
# steps_per_epoch_tr = lw_gen_tr.stp_per_epoch

##### load validation and make generator

sbj_n_s = range(1,11)
str_n_s = [2] 

lbl_vl = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
pickle.dump(lbl_vl, open("fullbody_lbl_vl.pkl", "w"))
print('val images loaded with shape: ',lbl_vl.shape)

img_vl = create_img_dataset(img_path, lbl_vl.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
pickle.dump(img_vl, open("fullbody_img_vl.pkl", "w"))
print('val labels loaded with shape: ',img_vl.shape)

# lw_gen_vl = light_generator(img_vl,lbl_vl,seq_len,batch_size)

# steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

##### Setting callbacks

# save_name = save_path+'/conv_3D_raw_face.{epoch:02d}-{val_loss:.2f}.hdf5'

# bckup_callback = cb.ModelCheckpoint(save_name, 
#                                     monitor='val_loss', 
#                                     verbose=0, 
#                                     save_best_only=True, 
#                                     save_weights_only=False, 
#                                     mode='auto', 
#                                     period=1)

# stop_callback = cb.EarlyStopping(monitor='val_loss', patience=5)

# callbacks_list = [
    
#     bckup_callback,
#     stop_callback,
#     cb.TensorBoard(log_dir="logs/" + day_time)
# ]

# m.fit_generator(   lw_gen_tr.generate(),
#                    steps_per_epoch=steps_per_epoch_tr, 
#                    epochs=epochs,
#                    callbacks=callbacks_list,
#                    validation_data = lw_gen_vl.generate(),
#                    validation_steps = steps_per_epoch_vl,
#                    shuffle=True)
