from model import conv_3d_id_model
from utils import *

import keras.backend as K
from keras import optimizers
#### load validation data

lbl_vl = np.load("../matrices/face_lbl_vl_small.npy", mmap_mode='r').reshape(-1,1)
print("val labels loaded with shape:", lbl_vl.shape)

video_valid_x = np.load("../matrices/face_img_vl_small.npy", mmap_mode='r')
print("val image loaded with shape:", video_valid_x.shape)

##### evaluate validation data

# change parameters depending on model
m = conv_3d_id_model(seq_len=10,img_x=48,img_y=48,ch_n=1).create()

opti = optimizers.Adam(lr=0.0001)
# change this filepath
m.load_weights('/Users/julialu/Downloads/bestmodels/face_small_new_generator.16-0.86.hdf5')
m.compile(loss='mean_squared_error', optimizer=opti)

lw_gen_vl = light_generator(video_valid_x,lbl_vl,seq_len=10,batch_size=64)
steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

#predictions_val = m.predict_generator(video_valid_x.generate(), batch_size=64, verbose=1)

predictions_val = m.predict_generator(lw_gen_vl.generate_no_shuffle(),
    steps=lw_gen_vl.stp_per_epoch)

print ("shape of predictions:", len(predictions_val))

X = predictions_val
#.reshape(predictions_val.shape[0],1)
np.save('prediction_face.npy', X)
Xcsv = np.load('prediction_face.npy')
np.savetxt('New_Generator_Results.csv', Xcsv, delimiter=",")



# length of the video (minus seq len-1)
preds = { 'valence': predict_text[start:end] }

df = DataFrame(preds, columns= ['valence'])