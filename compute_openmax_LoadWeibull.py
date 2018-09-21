# -*- coding: utf-8 -*-

###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat,savemat
import multiprocessing as mp


from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull
import numpy as np
try:
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()


#---------------------------------------------------------------------------------
# params and configuratoins
NCHANNELS = 10
NCLASSES = 1000
ALPHA_RANK = 10
WEIBULL_TAIL_SIZE = 20

#---------------------------------------------------------------------------------
def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax
    
    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class
    
    """
    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            channel_scores += [sp.exp(openmax_fc8[channel, category])]
                    
        total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
        prob_scores += [channel_scores/total_denominator ]
        prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :]))/total_denominator]
        
    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis = 0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    assert len(modified_scores) == 1001
    return modified_scores

#---------------------------------------------------------------------------------
def recalibrate_scores(weibull_model, labellist, imgarr,
                       layer = 'fc8', alpharank = 10, distance_type = 'eucos'):
    """ 
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained from weibull_tailfitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture
    
    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from caffe architecture. Function returns them for the sake 
    of convienence)

    """
    
    imglayer = imgarr[layer]
    ranked_list = imgarr['scores'].argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = sp.zeros(1000)
    for i in range(len(alpha_weights)):
        # print(ranked_list[i])
        # print(alpha_weights[i])
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        count = 0
        for categoryid in range(NCLASSES):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(labellist[categoryid], weibull_model, distance_type = distance_type)
            channel_distance = compute_distance(channel_scores, channel, category_weibull[0],
                                                distance_type = distance_type)

            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            modified_fc8_score = channel_scores[categoryid] * ( 1 - wscore*ranked_alpha[categoryid] )
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score ]

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = imgarr['scores'].ravel() 
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)


def Calweibull(outputFolder,listFiles,labellist):
    with open("weibul.pic", 'rb') as f:
        weibull_model = pickle.load(f)

    print("Completed Weibull fitting on %s models" % len(weibull_model.keys()))

    for sigleFile in listFiles:
        sigleFile = sigleFile.split(",")
        imagePath = sigleFile[0]
        classImage = sigleFile[1]
        __, ImageFileName = os.path.split(imagePath)
        FileDict = {}

        imgarr = loadmat(imagePath)
        openmax, softmax = recalibrate_scores(weibull_model, labellist, imgarr)

        FileDict["openMax"] = openmax
        FileDict["softMax"] = softmax
        FileDict["GTClass"] = classImage

        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        currentFileout = os.path.join(outputFolder,classImage)
        if not os.path.exists(currentFileout):
            os.mkdir(currentFileout)


        savePath_main = "{}/sp_detail_{}".format(currentFileout, ImageFileName)
        savemat(savePath_main, FileDict)
        print("FileName:{} Class {}".format(savePath_main, classImage))
        # print("File name {:s} Openmax Scores: {:d} Softmax :{:d} ".format(matFile,np.argmax(openmax)+1,Softmax_Predict+1))
        # print(openmax.shape, softmax.shape)

#---------------------------------------------------------------------------------
def main(args):

    synsetfname = args.synsetfname
    featurePath = args.FeaturePath
    outputFolder = args.outputFolder

    labellist = getlabellist(synsetfname)

    listFiles = extractDetailFile(featurePath)
    for idx in range(30):
        selected = [listFiles[i] for i in range(idx, len(listFiles), 30)]
        p = mp.Process(target=Calweibull, args=(outputFolder,selected,labellist))
        p.start()
        print("Finished {}".format(selected))



def extractDetailFile(featurePath):
    listFile = []
    for folderList in os.listdir(featurePath):
        currentpath = os.path.join(featurePath,folderList)
        for featureFile in os.listdir(currentpath):
            listFileFeature_detail = "{},{}".format(os.path.join(currentpath,featureFile),folderList)
            listFile += [listFileFeature_detail]
    return listFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website"
    )
    # ./FeatureCloseSet/n01443537/ILSVRC2012_val_00000172.mat
    parser.add_argument(
        "--FeaturePath",
        default='./FeatureCloseSet',
        help="This is a feature Path"
    )

    parser.add_argument(
        "--outputFolder",
        default='./SoftAndOpemMaxResult/',
        help="This File contained the detail of pathImage and image type"
    )


    args = parser.parse_args()
    main(args)
