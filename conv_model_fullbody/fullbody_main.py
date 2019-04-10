from fullbody_model import conv_3d_id_model
from utils import *

import keras.backend as K
from keras import optimizers


# full body paths
save_path = '/Users/julialu/cpsc490-senior-project/trained/conv_model_fullbody'
img_path = '/Users/julialu/cpsc490-senior-project/data/Training/FullBody/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '/Users/julialu/cpsc490-senior-project/data/Training/Annotations/Subject_{0}_Story_{1}.csv'

seq_len = 16
seq_len = 1
#img_x = 128
#img_y = 128
img_x = 48
img_y = 48
ch_n = 1

id_len = 16

batch_size = 128
epochs = 100

##### Creating model

K.clear_session()

m = conv_3d_id_model(seq_len, img_x, img_y, ch_n, id_len).create()
opti = optimizers.Adam(lr=0.0001)

m.compile(loss=ccc_error, optimizer=opti)

##### load training and make generator

#sbj_n_s = range(1,11)
# Story numbers that we are training with
# str_n_s = [1,4,5,8]
sbj_n_s = range(1,2)
str_n_s = [1]

lbl_tr = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
print('train labels loaded with shape: ',lbl_tr.shape) # (323675, 1))

img_tr = create_img_dataset(img_path, lbl_tr.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
print('train images loaded with shape: ',img_tr.shape) # (323675, 128, 128, 1) WHAT IT SHOULD BE

# ids_tr = make_id_vector(str_n_s,sbj_n_s,lbl_path) # what is this ? 
# print('train id loaded with shape: ',ids_tr.shape) # (323675, 1))

# lw_gen_tr = light_id_generator(img_tr[:],lbl_tr[:],ids_tr[:],seq_len,batch_size)
lw_gen_tr = light_generator(img_tr[:],lbl_tr[:],seq_len,batch_size)
steps_per_epoch_tr = lw_gen_tr.stp_per_epoch

##### load validation and make generator

#sbj_n_s = range(1,11)
sbj_n_s = range(1,2)
str_n_s = [2] 

lbl_vl = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for str_n in str_n_s for sbj_n in sbj_n_s]).reshape(-1,1)
print('valid images loaded with shape: ',img_tr.shape) # (95575, 128, 128, 1)

img_vl = create_img_dataset(img_path, lbl_vl.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
print('valid labels loaded with shape: ',lbl_tr.shape) # (95575, 1)

# ids_vl = make_id_vector(str_n_s,sbj_n_s,lbl_path)
# print('valid ids loaded with shape: ',ids_vl.shape) # (95575, 1)

#lw_gen_vl = light_id_generator(img_vl,lbl_vl,ids_vl,seq_len,batch_size)
lw_gen_vl = light_generator(img_vl,lbl_vl,seq_len,batch_size)

steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

##### Setting callbacks

save_name = save_path+'/conv_3D_raw_face.{epoch:02d}-{val_loss:.2f}.hdf5'

bckup_callback = cb.ModelCheckpoint(save_name, 
                                    monitor='val_loss', 
                                    verbose=0, 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='auto', 
                                    period=1)

stop_callback = cb.EarlyStopping(monitor='val_loss', patience=5)

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
