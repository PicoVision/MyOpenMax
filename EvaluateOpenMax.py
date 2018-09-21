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

def TravelFolder(SP_Result,mode):
    #return List of S_P_result File
    listFile = []
    if mode != "openset":
        print("CloseSet mode or Fooling Mode")
        for Folder in os.listdir(SP_Result):
            currentFolder = os.path.join(SP_Result, Folder)
            for sigleFile in os.listdir(currentFolder):
                currentFile = os.path.join(currentFolder, sigleFile)
                listFile += [currentFile]
    else:
        checkClassCredit  = {}
        for Folder in os.listdir(SP_Result):
            currentFolder = os.path.join(SP_Result, Folder)
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

def  EvaluateOpenMax(SP_Result,outputFile,labellist,classDict):
    #nameImage,GT,prob_sm,predict_sm,prob_om,predict_om
    with open(outputFile,'w') as oFile:
        Header_text = ",".join(["ImageName", "GT_Index",
                                "Prob_Of_Softmax", "SoftMaxPredict",
                                "Prob_Of_Openmax", "OpenMaxPredict"])
        oFile.write(Header_text + '\n')

        for sigleFile in SP_Result:
            SpFile = loadmat(sigleFile)
            ImageName = sigleFile
            openmaxScore = SpFile["openMax"][0]
            softmaxScore = SpFile["softMax"][0]

            GTClassImage = SpFile["GTClass"][0]
            if GTClassImage in classDict:
                GTClassIndex = classDict[GTClassImage]
            else:
                GTClassIndex = 1000

            predict_sm = np.argmax(softmaxScore)
            prob_sm = softmaxScore[predict_sm]

            predict_om = np.argmax(openmaxScore)
            prob_om = openmaxScore[predict_om]

            print(ImageName)
            Result_text = ",".join([ImageName,str(GTClassIndex),
                                    str(prob_sm),str(predict_sm),
                                    str(prob_om),str(predict_om)])
            oFile.write(Result_text+'\n')



# ---------------------------------------------------------------------------------
def main(args):
    synsetfname = args.synsetfname
    outfile = args.outfile
    SP_Result = args.SP_Result
    mode = args.Mode
    labellist = getlabellist(synsetfname)
    classDictmap = classDictMap(synsetfname)

    listFile = TravelFolder(SP_Result,mode)

    EvaluateOpenMax(listFile,outfile,labellist,classDictmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website default (<synset_words_caffe_ILSVRC12.txt>)"
    )
    parser.add_argument("--outfile",default="resultAll.txt",help="That is a threshole default 0")
    parser.add_argument(
        "--SP_Result",
        default='./S_P_CloseSet/',
        help="Soft_OpenMax result default <./S_P_CloseSet/>"
    )
    parser.add_argument("--Mode",default="CloseSet",
                        help="Mode CloseSet Openset FoolingImage default <CloseSet>")

    args = parser.parse_args()
    main(args)
