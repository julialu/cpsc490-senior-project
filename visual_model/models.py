from tensorflow.keras.layers import GRU, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, Dense, Dropout, Flatten, LSTM, Reshape, TimeDistributed, Input, concatenate, BatchNormalization
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import regularizers

class alphacity_conv3d():
  
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    
  def create(self):
    
    main_input = Input(shape=(self.seq_len,self.img_x,self.img_y,self.ch_n), name='main_input')

    layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(main_input)
    layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Flatten()(layer)
    layer = Dense(512,activation='relu', name='conv_out')(layer)
    layer = Dense(128,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    layer = Dense(32,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    reg_out = Dense(1,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in

class alphacity_conv3d_reduced():
  
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    
  def create(self):
    
    main_input = Input(shape=(self.seq_len,self.img_x,self.img_y,self.ch_n), name='main_input')

    layer = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(main_input)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(layer)
    layer = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer) 

    layer = Flatten()(layer)
    layer = Dense(512,activation='relu', name='conv_out')(layer)
    layer = Dense(128,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    layer = Dense(32,activation='relu')(layer)
    layer = Dropout(0.6)(layer)
    reg_out = Dense(1,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in

class cnn_facial_emotion():
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n

  def create(self):
    
    main_input = Input(shape=(self.seq_len,self.img_x,self.img_y,self.ch_n), name='main_input')

    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(main_input)
    layer = TimeDistributed(BatchNormalization())(layer)
    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
     

    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    
    
    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    
   
    layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)

    layer = TimeDistributed(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(Dropout(0.45))(layer)
    layer = TimeDistributed(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(Dropout(0.6))(layer)
    layer = Flatten()(layer)
    reg_out = Dense(self.seq_len,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in


class alphacity_timedistributed():
  
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    
  def create(self):
    
    main_input = Input(shape=(self.seq_len,self.img_x,self.img_y,self.ch_n), name='main_input')

    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(main_input)
    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    
    layer = TimeDistributed(Dense(512,activation='relu'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    layer = TimeDistributed(Dropout(0.6))(layer)
    layer = TimeDistributed(Dense(128,activation='relu'))(layer)
    layer = TimeDistributed(Dropout(0.6))(layer)
    layer = TimeDistributed(Dense(32,activation='relu'))(layer)
    layer = Flatten()(layer)
    reg_out = Dense(self.seq_len,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in

class cnn_GRU():
  
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    
  def create(self):
    
    main_input = Input(shape=(self.seq_len, self.img_x, self.img_y, self.ch_n), name='main_input')
    
    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(main_input)
    layer = TimeDistributed(Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    
    layer = TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Flatten())(layer)
    layer = TimeDistributed(Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(layer)
    layer = TimeDistributed(Dropout(0.2))(layer)
    layer = TimeDistributed(Flatten())(layer)
    layer = GRU(128)(layer)
    layer = Flatten()(layer)
    reg_out = Dense(self.seq_len,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in

class cnn_lstm():
  
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n
    
  def create(self):
    
    main_input = Input(shape=(self.seq_len, self.img_x, self.img_y, self.ch_n), name='main_input')
    
    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(main_input)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 
    
    layer = TimeDistributed(Dense(512 ,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(layer)

    layer = TimeDistributed(Flatten())(layer)

    layer = LSTM(128, return_sequences=True)(layer)
    layer = Flatten()(layer)
    reg_out = Dense(self.seq_len,activation='linear',name='reg_out')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[main_input], outputs=[reg_out])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in

class alphacity_timedistributed_reduced():
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n

  def create(self):
    video_input = Input(shape=(self.seq_len, self.img_x, self.img_y, self.ch_n), name='video_input')

    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(video_input)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(3, 3), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Flatten())(layer)
    layer = TimeDistributed(Dense(256, activation='relu', name='conv_out'))(layer)
    layer = TimeDistributed(Dense(32, activation='relu'))(layer)
    layer = Flatten()(layer)
    video_features = Dense(self.seq_len,activation='linear', name='video_features')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[video_input], outputs=[video_features])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_in


class tiny_dark_net():
  def __init__(self,seq_len,img_x,img_y,ch_n):
    
    self.seq_len = seq_len
    self.img_x = img_x
    self.img_y = img_y
    self.ch_n = ch_n

  def create(self):
    video_input = Input(shape=(self.seq_len, self.img_x, self.img_y, self.ch_n), name='video_input')

    layer = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))(video_input)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(16, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(16, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(Conv2D(1000, kernel_size=(1, 1), activation='relu', padding='same'))(layer)
    layer = TimeDistributed(BatchNormalization())(layer) 

    layer = TimeDistributed(Flatten())(layer)
    layer = TimeDistributed(Dense(1024,activation='relu', name='conv_out'))(layer)
    layer = TimeDistributed(Dropout(0.5))(layer)
    layer = Flatten()(layer)
    video_features = Dense(self.seq_len,activation='linear', name='video_features')(layer)

    reg_conv_3d_model_double_in = Model(inputs=[video_input], outputs=[video_features])
    reg_conv_3d_model_double_in.summary()
    return reg_conv_3d_model_double_ins