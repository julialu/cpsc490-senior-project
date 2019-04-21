from pandas import DataFrame
import numpy as np

'''
Converts predictions into CSV files. Make sure there is a file called all_results.csv
that saves all the outputs of your model
'''

# change this filepath to where the results of your model output is stored
predict_text = np.loadtxt('../visual_model/New_Generator_Results.csv')

# change these to which every story/subject is in the validation data and what the
# sequence length is for your model
sbj_n = range(1,4)
str_n = [1]
seq_len = 10

csv_path = 'Subject_{0}_Story_{1}.csv'
labels_csv_path = '../dataset/Validation/Annotations_Reduced/'

# indices into all predictions
start = 0
end = 0

print("total frames predicted:", len(predict_text))

for subject in sbj_n:
	for story in str_n:
		file_path = csv_path.format(subject, story)
		# count number of frames in original 
		num_frames = len(np.loadtxt(labels_csv_path + file_path, skiprows=1))
		print("num frames in validation for subject/story number:", num_frames, subject, story)

		# extract correct number of frames from predict_text
		end = end + num_frames - (seq_len - 1)
		frames = predict_text[start:end]
		print("num frames extracted from all predictions", len(frames))

		# pad with seq_len - 1 (possibly change the zero to 
		zeros = [0] * (seq_len - 1)
		frames = zeros + list(frames)
		print("num frames of prediction:", len(frames))

		# input into new csv file
		preds = { 'valence': frames }
		df = DataFrame(preds, columns= ['valence'])

		# change this folder for different models
		df.to_csv('visual_model/' + file_path, index=None, header=True)

		start = end