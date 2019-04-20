import numpy as np
import loadconfig
import os
import pandas
import ConfigParser
import essentia.standard as ess
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from scipy.signal import filtfilt, butter
import utilities_func as uf
import utilities_func as uf
from calculateCCC import ccc2
import feat_analysis2 as fa

#load configuration file
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')

batch_size = 128

# determined in preprocessing, NOT hyperparameter
frames_per_annotation = 4

#load classification model and latent extractor
print 'Loading model...'
valence_model = load_model('../models/audio_model.hdf5', custom_objects={'ccc_error':uf.ccc_error})
print 'Model successfully loaded'

print 'Loading dataset...'
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

#load datasets
# speech_train_x = np.load(SPEECH_TRAIN_PRED)
train_target = np.load(SPEECH_TRAIN_TARGET)
speech_valid_x = np.load(SPEECH_VALID_PRED)
validation_target = np.load(VALIDATION_TARGET)

n = 50000
speech_valid_x = speech_valid_x[:n*frames_per_annotation]
validation_target = validation_target[:n]

audio_gen_val = uf.audio_generator(speech_valid_x, validation_target, SEQ_LENGTH, batch_size, frames_per_annotation)
print 'Dataset successfully loaded'

print 'Getting predictions...'
predictions = valence_model.predict_generator(audio_gen_val.generate_no_shuffle(),
    steps=audio_gen_val.stp_per_epoch)

predictions = predictions.reshape(predictions.shape[0])

# apply f_trick
ann_folder = '../dataset/Training/Annotations'
target_mean, target_std = uf.find_mean_std(ann_folder)
predictions = uf.f_trick(predictions, target_mean, target_std)

#apply butterworth filter
b, a = butter(3, 0.01, 'low')
predictions = filtfilt(b, a, predictions)

print predictions
print validation_target

ccc = ccc2(predictions, validation_target[15:])  #compute ccc
print "CCC = " + str(ccc)

def predict_datapoint(input_sound, input_annotation):
    '''
    loads one audio file and predicts its coutinuous valence

    '''
    sr, samples = uf.wavread(input_sound)  #load
    e_samples = uf.preemphasis(samples, sr)  #apply preemphasis
    predictors = fa.extract_features(e_samples)  #compute power law spectrum
    #normalize by training mean and std
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    #load target
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target,(target.shape[0]))
    final_pred = []
    #compute prediction until last frame
    start = 0
    while start < (len(target)-SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
        #predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1], 1)

        prediction = valence_model.predict(predictors_temp)
        for i in range(prediction.shape[1]):
            final_pred.append(prediction[0][i])
        perc = int(float(start)/(len(target)-SEQ_LENGTH) * 100)
        print "Computing prediction: " + str(perc) + "%"
        start += 1

    final_pred = np.array(final_pred)



    '''
    #compute best prediction shift
    shifted_cccs = []
    time = np.add(1,range(200))
    print "Computing best optimization parameters"
    for i in time:
        t = target.copy()
        p = final_pred.copy()
        t = t[i:]
        p = p[:-i]
        #print t.shape, p.shape

        temp_ccc = ccc2(t, p)
        shifted_cccs.append(temp_ccc)


    best_shift = np.argmax(shifted_cccs)
    best_ccc = np.max(shifted_cccs)
    if best_shift > 0:
        best_target = target[best_shift:]
        best_pred = final_pred[:-best_shift]
    else:
        best_target = target
        best_pred = final_pred
    #print 'LEN BEST PRED: ' + str(len(best_pred))

    #compute best parameters for the filter
    test_freqs = []
    test_orders = []
    test_cccs = []
    freqs = np.arange(0.01,0.95,0.01)
    orders = np.arange(1,10,1)
    print "Finding best optimization parameters..."
    for freq in freqs:
        for order in orders:
            test_signal = best_pred.copy()
            b, a = butter(order, freq, 'low')
            filtered = filtfilt(b, a, test_signal)
            temp_ccc = ccc2(best_target, filtered)
            test_freqs.append(freq)
            test_orders.append(order)
            test_cccs.append(temp_ccc)
    best_filter = np.argmax(test_cccs)
    best_order = test_orders[best_filter]
    best_freq = test_freqs[best_filter]
    '''
    #POSTPROCESSING
    #normalize between -1 and 1
    # final_pred = np.multiply(final_pred, 2.)
    # final_pred = np.subtract(final_pred, 1.)

    #apply f_trick
    # ann_folder = '../dataset/Training/Annotations'
    # target_mean, target_std = uf.find_mean_std(ann_folder)
    # final_pred = uf.f_trick(final_pred, target_mean, target_std)

    # #apply butterworth filter
    # b, a = butter(3, 0.01, 'low')
    # final_pred = filtfilt(b, a, final_pred)

    print final_pred
    ccc = ccc2(final_pred, target)  #compute ccc
    print "CCC = " + str(ccc)

    '''
    plt.plot(target)
    plt.plot(final_pred, alpha=0.7)
    plt.legend(['target','prediction'])
    plt.show()
    '''

    return ccc

def extract_LLD_datapoint(input_sound, input_annotation):
    '''
    load one audio file and compute the model's last
    latent dimension
    '''
    sr, samples = uf.wavread(input_sound)  #load
    e_samples = uf.preemphasis(samples, sr)  #apply preemphasis
    predictors = fa.extract_features(e_samples)  #compute power law spectrum
    #normalize by training mean and std
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    final_vec = np.array([])
    #load target
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target,(target.shape[0]))

    #compute last latent dim until last frame
    start = 0
    while start < (len(target)-SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
        features_temp = latent_extractor([predictors_temp])
        features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
        if final_vec.shape[0] == 0:
            final_vec = features_temp
        else:
            final_vec = np.concatenate((final_vec, features_temp), axis=0)
        print 'Progress: '+ str(int(100*(final_vec.shape[0] / float(len(target))))) + '%'
        start += SEQ_LENGTH
    #compute last latent dim for last frame
    predictors_temp = predictors[-int(SEQ_LENGTH*frames_per_annotation):]
    predictors_temp = predictors_temp.reshape(1,predictors_temp.shape[0], predictors_temp.shape[1])
    features_temp = latent_extractor([predictors_temp])
    features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
    missing_samples = len(target) - final_vec.shape[0]
    last_vec = features_temp[-missing_samples:]
    final_vec = np.concatenate((final_vec, last_vec), axis=0)

    return final_vec

def evaluate_all_data(sound_dir, annotation_dir):
    '''
    compute prediction and ccc for all validation set
    '''
    list = os.listdir(annotation_dir)
    list = list[:]
    ccc = []
    for datapoint in list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print 'Processing: ' + name
        sound_file = sound_dir + '/' + name +".mp4.wav"
        temp_ccc = predict_datapoint(sound_file, annotation_file)
        ccc.append(temp_ccc)
    ccc = np.array(ccc)
    mean_ccc = np.mean(ccc)
    min_ccc = np.min(ccc)
    max_ccc = np.max(ccc)

    print "Mean CCC = " + str(mean_ccc)
    print "Min CCC = " + str(min_ccc)
    print "Max CCC = " + str(max_ccc)

def extract_LLD_dataset(sound_dir, annotation_dir):
    '''
    compute last latent dimension for all dataset
    '''
    list = os.listdir(annotation_dir)
    list = list[:]
    for datapoint in list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print 'Processing: ' + name
        sound_file = sound_dir + '/' + name +".mp4.wav"
        lld = extract_LLD_datapoint(sound_file, annotation_file)
        output_filename = LLD_DIR + '/' + name + '.npy'
        np.save(output_filename, lld)
