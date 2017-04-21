#!/usr/bin/env python2

import time
import argparse
import cv2
import os
import pickle
import sys

from operator import itemgetter

import numpy as np
# np.set_printoptions(precision=2)
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
from sklearn.neighbors import KNeighborsClassifier

from openface_healper import OpenFaceAnotater

# max_faces_found = 5

def iterImgs(directory):
    assert directory is not None

    exts = [".jpg", ".jpeg", ".png"]

    images = dict()

    for subdir, dirs, files in os.walk(directory):
        for path in files:
            (imageClass, fName) = (os.path.basename(subdir), path)
            (imageName, ext) = os.path.splitext(fName)
            if ext.lower() in exts:
                if imageClass in images:
                    images[imageClass].append(os.path.join(subdir, fName))
                else:
                    images[imageClass] = [os.path.join(subdir, fName)]
    return images

class TrainHandler(object):

    def __init__(self, args, openface_trainer):
        self.args = args
        self.openface_trainer = openface_trainer

    def get_data(self):
        '''Walks through parent directory and fills self.data'''
        # print('Getting Index')
        # if not self.args.vip:
        #     index = {}
        #     index_path = os.path.join(self.args.input, 'index.txt')
        #     with open(index_path) as f:
        #         lines = f.read().splitlines()
        #         for line in lines:
        #             person_id, person_name = line.split(' ',1)
        #             index[person_id] = person_name

        print('Getting Data')
        imgs = iterImgs(self.args.input)
        print (imgs)
        print ('Number of People {}'.format(len(imgs)))
        features = []
        labels = []
        for label in imgs:
            num_faces_found = 0
            for img in imgs[label]:
                # if num_faces_found >= max_faces_found:
                #     break
                try:
                    bgrImg = cv2.imread(img)
                    if bgrImg is None:
                        raise Exception("Unable to load image: {}".format(img))

                    r = self.openface_trainer.getRep(bgrImg, [], multiple=False, scale=None)[0]
                    # r = self.openface_trainer.getRep(bgrImg, multiple=False, scale=0.375)[0]
                    rep = r[1] #.reshape(1, -1)
                    # rep = r[1].reshape(1,-1)
                    features.append(rep)
                    # if self.args.vip:
                    labels.append(label)
                    print (label,img)
                    # else:
                        # labels.append(index[label])
                        # print (index[label],img)
                    num_faces_found += 1
                except Exception as e:
                    print str(e)
                    # pass
        print ('Created features and labels')
        return features,labels

    def train(self):
        '''Trains and saves classifier'''
        print 'Start Trainning'
        features,labels = self.get_data()
        print 'Feature shape',len(features)
        # print("labels", labels)
        # print "features",features
        le = LabelEncoder().fit(labels)
        print le
        labelsNum = le.transform(labels)
        print labelsNum
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        clf = KNeighborsClassifier(n_neighbors=1)

        # if args.ldaDim > 0:
        #     clf_final = clf
        #     clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),('clf', clf_final)])

        clf.fit(features,labelsNum)

        for rep,label in zip(features,labels):
            dist,ind = clf.kneighbors(rep.reshape(1,-1))
            # print dist[0]
            # print ind[0]
            nn_index = ind[0][0]
            nn_label = labels[nn_index]
            print 'nn index',nn_index
            print 'nearest neigbor',nn_label
            print 'actual person = ',label
            print 'nearest neighbor distance',dist[0][0]
            

        fName = "{}/classifier.pkl".format(self.args.input)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)

class TrainArgParser(argparse.ArgumentParser):
    """Argument parser class"""

    def set(self):
        """Setup parser"""

        self.add_argument(
            '-i', '--input',
            action='store',
            default=os.environ.get('IDIR_PATH', None),
            help='path to input dir (FILE_PATH)')
        self.add_argument(
            '--version',
            action='version',
            version='%(prog)s 0.0')
        self.add_argument(
           '--vip',
           action='store_true')

def main(argv = sys.argv):
    arg_parser = TrainArgParser(
        prog='openface_trainer',
        description='OpenFace Trainer')
    arg_parser.set()
    args, argv = arg_parser.parse_known_args(argv)

    # global openface_trainer
    openface_trainer = OpenFaceAnotater(argv)

    train_handler = TrainHandler(args, openface_trainer)
    train_handler.train()

if __name__ == '__main__':
    main()
