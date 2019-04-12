from utils import *

import pickle 

'''
Converts frames from video into numpy arrays and saves into 4 files: 
- fullbody_lbl_tr.pkl
- fullbody_img_tr.pkl
- fullbody_lbl_vl.pkl
- fullbody_img_vl.pkl
'''

# paths to where images and labels are stored
img_path = '../dataset/Training/FullBody/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '../dataset/Training/Annotations/Subject_{0}_Story_{1}.csv'

# dimensions to resize images to
img_x = 48
img_y = 48
ch_n = 1

##### load training and make generator

# refers to the story and subject #s in video names
sbj_n_s = range(1,11)
str_n_s = [2,4,5,8]

lbl_tr = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for sbj_n in sbj_n_s for str_n in str_n_s])
pickle.dump(lbl_tr, open("../matrices/fullbody_lbl_tr.pkl", "w"))
print('train labels loaded with shape: ',lbl_tr.shape) 

img_tr = create_img_dataset(img_path, lbl_tr.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
pickle.dump(img_tr, open("../matrices/fullbody_img_tr.pkl", "w"))
print('train images loaded with shape: ',img_tr.shape) 


##### load validation and make generator

img_path = '../dataset/Validation/FullBody/Subject_{0}_Story_{1}/Subject_img/'
lbl_path = '../dataset/Validation/Annotations/Subject_{0}_Story_{1}.csv'

sbj_n_s = range(1,11)
str_n_s = [1] 

lbl_vl = np.concatenate([np.loadtxt(lbl_path.format(sbj_n,str_n),skiprows=1) for sbj_n in sbj_n_s for str_n in str_n_s])
pickle.dump(lbl_vl, open("../matrices/fullbody_lbl_vl.pkl", "w"))
print('val images loaded with shape: ',lbl_vl.shape)

img_vl = create_img_dataset(img_path, lbl_vl.shape[0],img_x,img_y,ch_n,str_n_s,sbj_n_s)
pickle.dump(img_vl, open("../matrices/fullbody_img_vl.pkl", "w"))
print('val labels loaded with shape: ',img_vl.shape)