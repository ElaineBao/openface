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

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
parser.add_argument("-i", "--image", help="Path to Image", required="True")
args = parser.parse_args()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
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
align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor) #init face mean and face key points for alignment
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

    '''
    start = time.time()
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    print "before,",args.imgDim
    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    print "after,",args.imgDim
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))
	'''
    start = time.time()
    alignedFace=cv2.resize(img,(args.imgDim,args.imgDim),interpolation=cv2.INTER_AREA)
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

num=range(len(args.imgs))
print args.imgs
imgs_rep=np.zeros((len(args.imgs),128))
for index in num:
	path=args.imgs[index]
	rep=getRep(path)
	imgs_rep[index]=rep	


#cluster by hierarchy clustering
disMat = sch.distance.pdist(imgs_rep) 
#print disMat
Z=sch.linkage(disMat,method='average')
P=sch.dendrogram(Z)
plt.savefig('plot_dendrogram.png')
cluster= sch.fcluster(Z, 0.5, 'inconsistent')
#fcluster=sch.fclusterdata(imgs_rep, 0.9, criterion='inconsistent', metric='euclidean', depth=2, method='average', R=None)
print "Original cluster by hierarchy clustering:\n",cluster

data=whiten(imgs_rep)
'''select k
centroid=[]
distortion=[]
for k in range(len(args.imgs)/3):
    k=k+1
    outp=kmeans(data,k)
    centroid.append(outp[0])
    distortion.append(outp[1])
print "distortion:",distortion
select_k=distortion.index(min(distortion))
print "select k:",k
print centroid[k]
'''
centroid=kmeans(data,max(cluster))[0]
#print type(data),type(centroid)
label=vq(data,centroid)[0]
'''[centroid,label]=kmeans2(data,4)'''
print "Final clustering by k-means:\n",label
