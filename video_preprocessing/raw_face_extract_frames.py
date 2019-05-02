# Code from OMG Empathy Challenge repository

import cv2
import os
import dlib

import subprocess
import shutil
from shutil import copyfile
import sys

from skimage.transform import resize
from skimage.color import rgb2gray
from numpy import reshape

'''
extracts frames from subject in videos 
and resizes to 48x48 bw images
'''

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def extractFramesFromVideo(path,savePath, faceDetectorPrecision, size):
    videos = os.listdir(path + "/")

    for video in videos:

        # if '.DS_Store' in videos:
        #     videos.remove('.DS_Store')

        videoPath = path + "/" + video
        print "- Processing Video:", videoPath + " ..."
        detector = dlib.get_frontal_face_detector()
        dataX = []

        copyTarget = "/Users/julialu/cpsc490-senior-project/dataset/Test/Videos/clip1.mp4"

        print "--- Copying file:", videoPath + " ..."
        copyfile(videoPath, copyTarget)
        cap = cv2.VideoCapture(copyTarget)

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        numberOfImages = 0
        check = True
        flag = True
        imageNumber = 0
        lastImageWithFaceDetected = 0
        print "- Extracting Faces:", str(totalFrames) + " Frames ..."

        savePathActor = savePath + "/" + video + "/Actor/"
        savePathSubject = savePath + "/" + video + "/Subject/"

        if not os.path.exists(savePathActor):
            os.makedirs(savePathActor)
            os.makedirs(savePathSubject)
            while (check):
                    check, img = cap.read()
                    if img is not None:

                        # Extract Subject Face
                        imageSubject = img[0:720, 1080:2560]
                        if lastImageWithFaceDetected == 0 or lastImageWithFaceDetected > faceDetectorPrecision:
                            dets = detector(imageSubject, 1)
                            lastImageWithFaceDetected = 0

                            if not len(dets) == 0:
                                oldDetsSubject = dets
                        else:
                            dets = oldDetsSubject

                        #try:
                        if not len(dets) == 0:
                            for i, d in enumerate(dets):
                                croped = imageSubject[d.top():d.bottom(), d.left():d.right()]
                                croped = resize(croped, (size, size))
                                croped = rgb2gray(croped).reshape(size, size,1)
                                cv2.imwrite(savePathSubject + "/%d.png" % imageNumber, croped*255)
               
                        else:
                            cv2.imwrite(savePathActor + "/%d.png" % imageNumber, imageSubject)

                        # except:
                        #     print "------error!"

                        imageNumber = imageNumber + 1
                        lastImageWithFaceDetected = lastImageWithFaceDetected + 1
                        progressBar(imageNumber, totalFrames)


if __name__ == "__main__":


     #Path where the videos are
    path = "../dataset/Test/Videos"

     #Path where the faces will be saved
    savePath ="../dataset/Test/Faces"

    # If 1, the face detector will act upon each of the frames. If 1000, the face detector update its position every 100 frames.
    faceDetectorPrecision = 9

    size = 48

    detector = dlib.get_frontal_face_detector()

    extractFramesFromVideo(path, savePath, faceDetectorPrecision, size)
 