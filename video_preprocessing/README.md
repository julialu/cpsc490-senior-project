## Video Preprocessing

### `fullbody_extract_frames.py`

This file extracts frames from each video in the dataset and crops the full-body of the subject and actor
(from: https://github.com/omg-challenge-alpha/omg_challenge2018_submission_code/)

### `raw_face_extract_frames.py`

This file runs a face detector on the videos and extracts frames with the subject's face
(from: https://github.com/omg-challenge-alpha/omg_challenge2018_submission_code/tree/master/fullbody)

### `preprocess_frames.py`

This file can be run after extracting frames. It creates it converts images (and labels) into `npy` 
arrays and saves them to file