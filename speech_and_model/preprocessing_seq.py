import numpy as np
import os
import utilities_func as uf
import feat_analysis2 as fa
import pandas
import loadconfig
import ConfigParser

#load configuration file
config = loadconfig.load()
cfg = ConfigParser.ConfigParser()
cfg.read(config)

SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
SEQ_OVERLAP = cfg.getfloat('preprocessing', 'sequence_overlap')

SOUND_FOLDER_T = cfg.get('preprocessing', 'input_audio_folder_t')
ANNOTATION_FOLDER_T = cfg.get('preprocessing', 'input_annotation_folder_t')
OUTPUT_PREDICTORS_MATRIX_T = cfg.get('preprocessing', 'output_predictors_matrix_t')
OUTPUT_TARGET_MATRIX_T = cfg.get('preprocessing', 'output_target_matrix_t')

SOUND_FOLDER_V = cfg.get('preprocessing', 'input_audio_folder_v')
ANNOTATION_FOLDER_V = cfg.get('preprocessing', 'input_annotation_folder_v')
OUTPUT_PREDICTORS_MATRIX_V = cfg.get('preprocessing', 'output_predictors_matrix_v')
OUTPUT_TARGET_MATRIX_V = cfg.get('preprocessing', 'output_target_matrix_v')

SOUND_FOLDER_TEST = cfg.get('preprocessing', 'input_audio_folder_test')
ANNOTATION_FOLDER_TEST = cfg.get('preprocessing', 'input_annotation_folder_test')
OUTPUT_PREDICTORS_MATRIX_TEST = cfg.get('preprocessing', 'output_predictors_matrix_test')
OUTPUT_TARGET_MATRIX_TEST = cfg.get('preprocessing', 'output_target_matrix_test')

TARGET_SUBJECT = cfg.get('preprocessing', 'target_subject')
TARGET_STORY = cfg.get('preprocessing', 'target_story')
TARGET_DELAY = cfg.getint('preprocessing', 'target_delay')
SR = cfg.getint('sampling', 'sr')
HOP_SIZE = cfg.getint('stft', 'hop_size')

fps = 25  #annotations per second
hop_annotation = SR /fps
frames_per_annotation = hop_annotation/float(HOP_SIZE)
#frames_per_annotation = int(np.round(frames_per_annotation))
'''
reminder = frames_per_annotation % 1

if reminder != 0.:
    raise ValueError('Hop size must be a divider of annotation hop (640)')
else:
    frames_per_annotation = int(frames_per_annotation)
'''
frames_delay = int(TARGET_DELAY * frames_per_annotation)


def filter_items(contents_list, target_subj='all', target_story='all'):
    '''
    return a list with filenames containing only a desired subject or story

    '''
    target_subj = str(target_subj)
    target_story = str(target_story)
    final_list = []
    if target_subj == 'all':
        subj_list = contents_list
    else:
        subj_list = []
        for file in contents_list:
            subj = file.split('Subject_')[1]
            subj = subj.split('_')[0]
            if subj == target_subj:
                subj_list.append(file)
    if target_story == 'all':
        story_list = contents_list
    else:
        story_list = []
        for file in contents_list:
            story = file.split('Story_')[1]
            story = story.split('.')[0]
            if story == target_story:
                story_list.append(file)
                print 'iiiiii'
    for subj in subj_list:
        if subj in story_list:
            final_list.append(subj)

    return final_list

def preprocess_datapoint(input_sound, input_annotation):
    '''
    generate predictors (stft) and target (valence sequence)
    of one sound file from the OMG dataset
    '''
    sr, samples = uf.wavread(input_sound)  #read audio
    e_samples = uf.preemphasis(samples, sr)  #apply preemphasis
    feats = fa.extract_features(e_samples)  #extract features
    annotation = pandas.read_csv(input_annotation)  #read annotations
    annotation = annotation.values
    annotation = np.reshape(annotation, annotation.shape[0])
    annotated_frames = int(len(annotation) * frames_per_annotation)
    feats = feats[:annotated_frames]  #discard non annotated final frames
    annotation = annotation[TARGET_DELAY:]  #shift back annotations by target_delay
    feats2 = feats[:-frames_delay]

    return feats, annotation

def segment_datapoint(features, annotation, sequence_length, sequence_overlap):
    '''
    segment features and annotations of one long audio file
    into smaller matrices of length "sequence_length"
    and overlapped by "sequence_overlap"
    '''
    pointer = np.arange(0,len(annotation) - sequence_length + 1, 1, dtype='int')  #initail positions of segments
    predictors = []
    target = []
    #slice arrays and append datapoints to vectors
    for start in pointer:
        start_annotation = start
        stop_annotation = start + sequence_length
        start_features = int(start_annotation * frames_per_annotation)
        stop_features = int(stop_annotation * frames_per_annotation)
        #print start_annotation, stop_annotation, start_features, stop_features
        if stop_annotation <= len(annotation):
            temp_predictors = features[start_features:stop_features]
            temp_target = annotation[stop_annotation-1] # target is annotation at last feature
            predictors.append(temp_predictors)
            target.append(temp_target)
            #target.append(np.mean(temp_target))

    predictors = np.array(predictors)
    target = np.array(target)

    return predictors, target


def preprocess_dataset(sound_folder, annotation_folder, target_subject='all', target_story='all', mode='training'):
    '''
    build dataset numpy matrices:
    -predictors: contatining audio features
    -target: contatining correspective valence annotations
    both are NOT normalized
    datapoints order is randomly scrambled
    '''
    predictors = []
    target = []
    fileNameFormat = 'Subject_{0}_Story_{1}'
    sbj_n_s = range(1,11)
    str_n_s = [2,4,5,8]
    if mode == 'validation':
        str_n_s = [1]
    elif mode == 'test':
        str_n_s = [3, 6, 7]
    # filtered_list = filter_items(annotations, target_subject, target_story)
    num_sounds = len(str_n_s) * len(sbj_n_s)
    #process all files in folders
    index = 0
    for sbj_n in sbj_n_s:
        for str_n in str_n_s:
            name = fileNameFormat.format(sbj_n, str_n)
            print name
            annotation_file = annotation_folder + '/' + name + '.csv'
            sound_file = sound_folder + '/' + name + ".wav"  #get correspective sound
            long_predictors, long_target = preprocess_datapoint(sound_file, annotation_file)  #compute features
            # cut_predictors, cut_target = segment_datapoint(long_predictors, long_target,   #slice feature maps
            #                                                 SEQ_LENGTH, SEQ_OVERLAP)

            # predictors.append(cut_predictors)
            # target.append(cut_target)
            predictors.extend(long_predictors)
            target.extend(long_target)
            perc_progress = (index * 100) / num_sounds
            index += 1
            print "processed files: " + str(index) + " over " + str(num_sounds) + "  |  progress: " + str(perc_progress) + "%"

    # predictors = np.concatenate(predictors, axis=0)  #reshape arrays
    # target = np.concatenate(target, axis=0)
    #scramble datapoints order
    # shuffled_predictors = []
    # shuffled_target = []
    # num_datapoints = target.shape[0]
    # random_indices = range(num_datapoints)
    # np.random.shuffle(random_indices)
    # for i in random_indices:
    #     shuffled_predictors.append(predictors[i])
    #     shuffled_target.append(target[i])
    # shuffled_predictors = np.array(shuffled_predictors)
    # shuffled_target = np.array(shuffled_target)

    return np.array(predictors), np.array(target)

def build_matrices(output_predictors_matrix, output_target_matrix, sound_folder, annotation_folder, mode):
    '''
    build matrices and save numpy files
    '''
    predictors, target = preprocess_dataset(sound_folder, annotation_folder, TARGET_SUBJECT, TARGET_STORY, mode)

    np.save(output_predictors_matrix, predictors)
    np.save(output_target_matrix, target)
    print "Matrices saved succesfully"
    print 'predictors shape: ' + str(predictors.shape)
    print 'target shape: ' + str(target.shape)


if __name__ == '__main__':
    '''
    build training and validation matrices
    '''
    build_matrices(OUTPUT_PREDICTORS_MATRIX_T, OUTPUT_TARGET_MATRIX_T, SOUND_FOLDER_T, ANNOTATION_FOLDER_T, 'training')
    build_matrices(OUTPUT_PREDICTORS_MATRIX_V, OUTPUT_TARGET_MATRIX_V, SOUND_FOLDER_V, ANNOTATION_FOLDER_V, 'validation')
    build_matrices(OUTPUT_PREDICTORS_MATRIX_TEST, OUTPUT_TARGET_MATRIX_TEST, SOUND_FOLDER_TEST, ANNOTATION_FOLDER_TEST, 'test')
