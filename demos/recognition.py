#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()
import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))
import math
import openface
import openface.helper
from openface.data import iterImgs

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, help="Input a query image.")
parser.add_argument('--dbpath',type=str, default=os.path.join(fileDir, '..', 'database') , help="recognition database.")
parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--dlibRoot', type=str,
                    default=os.path.expanduser(
                        "~/src/dlib-18.16/python_examples"),
                    help="dlib directory with the dlib.so Python library.")
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

sys.path.append(args.dlibRoot)
import dlib

from openface.alignment import NaiveDlib  # Depends on dlib.
if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    img = cv2.imread(imgPath)
    if img is None:
        raise Exception("Unable to load image: {}".format(imgPath))       
    if args.verbose:
        print("  + Original size: {}".format(img.shape))

    
    start = time.time()
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None: #find no face in the img, maybe due to large face.
        #raise Exception("Unable to find a face: {}".format(imgPath))
        alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)
    else:
        img_crop = img[bb.top():bb.bottom(),bb.left():bb.right(),:]
        alignedFace=cv2.resize(img_crop,(args.imgDim,args.imgDim))
	
	'''
    start = time.time()
    print "before,",args.imgDim
    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    print "after,",args.imgDim
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))
	'''
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))	
    start = time.time()
    rep = net.forwardImage(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

#pair=itertools.combinations(args.imgs, 2)
#print "paths:",args.imgs #paths of all the images
print("initialize database representations")
start=time.time()
identities=os.listdir(args.dbpath) #it's a list
numid=len(identities) #number of persons in the database
print args.dbpath,identities,type(identities)
print ("number of persons in the database:",numid)
dbrep=np.zeros([128,1])
dbid=[]
index=0
for person in identities:
    personpath=os.path.join(args.dbpath,person) #the path to the dictionary of each person
    items=os.listdir(personpath) #it's a list
    #print itempath[0]
    itemnum=len(items)
    for item in items:
        itempath=os.path.join(personpath,item)
        index=index+1
        print index, itempath
        if index==182:
        	#raise Exception("the end")
        	rep=getRep(itempath)

        rep=rep.reshape((128,1))
        #print dbrep.shape,rep.shape,type(rep)
        print 'dbrep is appended:',dbrep.shape,type(dbrep)
        dbrep=np.append(dbrep,rep,axis=1)
        dbid.append(person)
dbrep=dbrep[:,1:]
print("saving database")
np.savez('dbinfo',dbrep,dbid)

    #print personpath,type(itempath),itemnum
'''
    rarray=np.random.random(size=10)*itemnum #randomly generate 10 images to describe the person
    #print rarray,math.floor(rarray[0])
    for item in rarray:
    	imgPath=itempath[int(math.floor(item))]
    	print imgPath
'''


'''
num=range(len(args.imgs)) # coding for each image, 0,1,2,...
pair=itertools.combinations(num, 2)
pair=list(pair)
#print pp[0],pp[0][1]
#print pp
simMat=np.zeros((len(args.imgs),len(args.imgs)))+100
#print simMat
num_pair=len(pair)
while num_pair>0:
	num_pair=num_pair-1
	index1=pair[num_pair][0]
	index2=pair[num_pair][1]
	print "index:",index1,index2
	path1=args.imgs[index1]
	path2=args.imgs[index2]
	rep1=getRep(path1)
	rep2=getRep(path2)
	d=rep1 - rep2
	l2_distance=np.dot(d,d)
	simMat[index1,index2]=l2_distance
	simMat[index2,index1]=l2_distance
	print simMat
    #print("Comparing {} with {}.".format(img1, img2))
    #print("  + Squared l2 distance between representations: {}".format(l2_distance))
'''