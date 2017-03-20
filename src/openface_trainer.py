﻿import time
import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from openface.data import iterImgs

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models') #models is 1 level up
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

class OpenFaceModel(Object):
    '''Trains a face recognition model from image features'''
    def getRep(self,imgPath, multiple=False):
        '''Takes image path as input to give a list of feature vectors, each feature vector corresponds to a face in image.'''
        start = time.time()
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))
        if args.verbose:
            print("Loading the image took {} seconds.".format(time.time() - start))

        start = time.time()

        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            raise Exception("Unable to find a face: {}".format(imgPath))
        if args.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))

        reps = []
        for bb in bbs:
            start = time.time()
            alignedFace = align.align(
                args.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                # raise Exception("Unable to align image: {}".format(imgPath))
                print("Unable to align image: {}".format(imgPath))
                continue
            if args.verbose:
                print("Alignment took {} seconds.".format(time.time() - start))
                print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

            start = time.time()
            rep = net.forward(alignedFace)
            if args.verbose:
                print("Neural network forward pass took {} seconds.".format(
                    time.time() - start))
        #Assumption only one face per photo. Because for a given feature representation we have only one label.
        return rep 

    def get_data(self):
        '''Walks through parent directory and fills self.data'''
        print('Getting Data')
        imgs = list(iterImgs(args.inputDir))
        print ('Number of images {}'.format(len(imgs)))
        features = []
        labels = []
        for imgObject in imgs:
            features.append(getRep(imgObject.path,False))
            labels.append(imgObject.cls)
        print ('Created features and labels')
        return features,labels

    def train(self):
        '''Trains and saves classifier'''
        features,labels = get_data()
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        clf = SVC(C=1, kernel='linear', probability=True)

        if args.ldaDim > 0:
            clf_final = clf
            clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),('clf', clf_final)])

        clf.fit(features,labelsNum)

        print (labels,labelsNum)

        for rep,label,labelnum in zip(features,labels,labelsNum):
			predictions = clf.predict_proba(rep).ravel()
	        maxI = np.argmax(predictions)
	        print ('predicted index = ',maxI)
	        person = le.inverse_transform(maxI)
	        print ('predicted person = ',person)
	        confidence = predictions[maxI]
	        print ('confidence =',confidence)
	        print ('actual index = ',labelnum)
	        print ('actual person = ',label)

        fName = "{}/classifier.pkl".format(fileDir)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('inputDir', type=str, help="Input image directory.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ldaDim', type=int, default=-1)

    #args, align and net are global to the script because they are in the main module
    args = parser.parse_args()
    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    OPM = OpenFaceModel()
    OPM.train()