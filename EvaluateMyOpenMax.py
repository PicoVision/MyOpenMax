# -*- coding: utf-8 -*-

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat, savemat
import multiprocessing as mp
import xlsxwriter


from openmax_utils import *
from evt_fitting import weibull_tailfitting, query_weibull
import numpy as np

def TravelFolder(featureData,mode):
    #return List of S_P_result File
    listFile = []
    if mode != "openset":
        print("CloseSet mode or Fooling Mode")
        for Folder in os.listdir(featureData):
            currentFolder = os.path.join(featureData, Folder)
            for sigleFile in os.listdir(currentFolder):
                currentFile = os.path.join(currentFolder, sigleFile)
                listFile += [currentFile]
    else:
        checkClassCredit  = {}
        for Folder in os.listdir(featureData):
            currentFolder = os.path.join(featureData, Folder)
            for sigleFile in os.listdir(currentFolder):
                classname = sigleFile.split("_")[2]
                if not classname in checkClassCredit:
                    checkClassCredit[classname] = 1
                else:
                    checkClassCredit[classname] += 1

                if checkClassCredit[classname] >50:
                    continue
                currentFile = os.path.join(currentFolder, sigleFile)
                listFile += [currentFile]
    print("Total File :{}".format(len(listFile)))
    return listFile

def  EvaluateMyOpenMax(SP_Result,outputFile,labellist,classDict):
    #nameImage,GT,prob_sm,predict_sm,prob_om,predict_om
    with open("Lecog_"+outputFile,'w') as oFile:
        #create Header File
        head = list(range(1000))
        Header_text = ",".join(map(str,head))
        Header_text = "ImageName,GT_Index,Prob_Of_Softmax, SoftMaxPredict,eak,pow3Ak,{:s}".format(Header_text)

        oFile.write(Header_text + '\n')

        for sigleFile in SP_Result:
            SpFile = loadmat(sigleFile)
            ImageName = sigleFile
            fc8_10 = SpFile["fc8"]
            scores = SpFile["scores"][0]

            GTClassImage = os.path.basename(os.path.dirname(SpFile["IMG_NAME"][0]))

            if GTClassImage in classDict:
                GTClassIndex = classDict[GTClassImage]
            else:
                GTClassIndex = 1000

            #shape fc8_10 -> [10,1000] reduce with average to [1000]
            fc8_10 = np.mean(fc8_10,axis=0)
            pow3_AK = np.power(np.sum(fc8_10,axis=0),3) #find power 3 of sum product
            eak = np.exp(np.sum(fc8_10,axis=0))     #find eak of sum product


            predict_sm = np.argmax(scores)  #get class index max
            prob_sm = scores[predict_sm]    #get prob of prediction


            print(ImageName)
            Result_text = ",".join([ImageName,str(GTClassIndex),
                                    str(prob_sm),str(predict_sm),
                                    str(eak),str(pow3_AK)])
            resultF8 = ",".join(map(str,fc8_10))
            Result_text = Result_text+","+resultF8

            oFile.write(Result_text+'\n')



# ---------------------------------------------------------------------------------
def main(args):
    synsetfname = args.synsetfname
    outfile = args.outfile
    featureData = args.featureData
    mode = args.Mode
    labellist = getlabellist(synsetfname)
    classDictmap = classDictMap(synsetfname)

    listFile = TravelFolder(featureData,mode)

    EvaluateMyOpenMax(listFile,outfile,labellist,classDictmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website default (<synset_words_caffe_ILSVRC12.txt>)"
    )
    parser.add_argument("--outfile",default="MyOpenresultAll.txt",help="That is a threshole default 0")
    parser.add_argument(
        "--featureData",
        default='./FeatureCloseSet/',
        help="Soft_OpenMax result default <./FeatureCloseSet/>"
    )
    parser.add_argument("--Mode",default="CloseSet",
                        help="Mode CloseSet Openset FoolingImage default <CloseSet>")

    args = parser.parse_args()
    main(args)
