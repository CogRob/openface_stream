#!/usr/bin/env python2

import time

start = time.time()

import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

modelDir = os.path.join('/root/openface', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

class OpenFaceArgParser(argparse.ArgumentParser):
    """Argument parser class"""

    def set(self):
        """Setup parser"""

        self.add_argument(
            '--dlibFacePredictor',
            type=str,
            help="Path to dlib's face predictor.",
            default=os.path.join(
                dlibModelDir,
                "shape_predictor_68_face_landmarks.dat"))
        self.add_argument(
            '--networkModel',
            type=str,
            help="Path to Torch network model.",
            default=os.path.join(
                openfaceModelDir,
                'nn4.small2.v1.t7'))
        self.add_argument(
            '--classifierModel',
            type=str,
            help="Path to Classifier Model.",
            default=os.path.join('/root/openface_stream/src','celeb_img_trained.pkl'))
        self.add_argument(
            '--imgDim',
            type=int,
            help="Default image dimension.",
            default=96)
        self.add_argument(
            '--cuda',
            action='store_true')
        self.add_argument(
            '--verbose',
            action='store_true')

class OpenFaceAnotater(object):

    def __init__(self, argv):
        self.arg_parser = OpenFaceArgParser(
            prog='openface',
            description='OpenFace')
        self.arg_parser.set()
        self.args, argv = self.arg_parser.parse_known_args(argv)
        if self.args.verbose:
            print("Argument parsing and import libraries took {} seconds.".format(
                time.time() - start))

        start = time.time()
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(
            self.args.networkModel,
            imgDim=self.args.imgDim,
            cuda=self.args.cuda)

        self.load(self.args.classifierModel)

        if self.args.verbose:
            print("Loading the dlib and OpenFace models took {} seconds.".format(
                time.time() - start))
            start = time.time()

    def load(self, classifierModel):
        with open(classifierModel, 'r') as f:
            (le, clf) = pickle.load(f)
        self.le = le
        self.clf = clf

    def predict(self, img, multiple=False):
        bgrImg = img
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        annotatedImg = np.copy(rgbImg)

        reps = self.getRep(img, multiple)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            bb = r[2] #bounding box
            landmarks = r[3] #landmarks
            start = time.time()
            predictions = self.clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = self.le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if self.args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(
                        person,
                        bbx,
                        confidence))
                else:
                    print("Predict {} with {:.2f} confidence.".format(
                        person,
                        confidence))
            if isinstance(self.clf, GMM):
                dist = np.linalg.norm(rep - self.clf.means_[maxI])
                if self.args.verbose:
                    print("  + Distance from the mean: {}".format(dist))

            #code for annotated bounding box
            bl = (bb.left(), bb.bottom())
            tr = (bb.right(), bb.top())
            cv2.rectangle(annotatedImg, bl, tr, color=(153, 255, 204),thickness=3)
            for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                cv2.circle(annotatedImg, center=landmarks[p], radius=3,
                               color=(102, 204, 255), thickness=-1)
            cv2.putText(annotatedImg, person, (bb.left(), bb.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(152, 255, 204), thickness=2)
        annotatedImgBgr = cv2.cvtColor(annotatedImg, cv2.COLOR_RGB2BGR)
        return annotatedImgBgr

    def getRep(self, rgbImg, multiple=False):
        start = time.time()

        if multiple:
            bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = self.align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            raise Exception("Unable to find a face")
        if self.args.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))

        reps = []
        for bb in bbs:
            start = time.time()
            alignedFace = self.align.align(
                self.args.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise Exception("Unable to align image")
            if args.verbose:
                print("Alignment took {} seconds.".format(time.time() - start))
                print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))
            landmarks = self.align.findLandmarks(rgbImg,bb)
            start = time.time()
            rep = self.net.forward(alignedFace)
            if self.args.verbose:
                print("Neural network forward pass took {} seconds.".format(
                    time.time() - start))
            reps.append((bb.center().x, rep, bb, landmarks)) #added the bounding box and landmarks
        sreps = sorted(reps, key=lambda x: x[0])
        return sreps
