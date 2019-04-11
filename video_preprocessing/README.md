## Video Preprocessing

### `fullbody_extract_frames.py`

This file extracts frames from each video in the dataset and crops the full-body of the subject and actor

### `fullbody_preprocess_frames.py`

This file can be run after `fullbody_extract_frames.py`. It creates 4 different `pkl` files containing labels and image numpy arrays for training and validation. 

Both of these files are taken from: https://github.com/omg-challenge-alpha/omg_challenge2018_submission_code/tree/master/fullbody
