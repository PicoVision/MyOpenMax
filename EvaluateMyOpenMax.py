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


def TravelFolder(featureData, mode):
    # return List of S_P_result File
    listFile = []
    if mode != "openset":
        print("CloseSet mode or Fooling Mode")
        for Folder in os.listdir(featureData):
            currentFolder = os.path.join(featureData, Folder)
            for sigleFile in os.listdir(currentFolder):
                currentFile = os.path.join(currentFolder, sigleFile)
                listFile += [currentFile]
    else:
        print("Openset Mode")
        checkClassCredit = {}
        for Folder in os.listdir(featureData):
            currentFolder = os.path.join(featureData, Folder)
            for sigleFile in os.listdir(currentFolder):
                print(sigleFile)
                classname = sigleFile.split("_")[0]
                if not classname in checkClassCredit:
                    checkClassCredit[classname] = 1
                else:
                    checkClassCredit[classname] += 1

                # if checkClassCredit[classname] >50:
                #     continue
                currentFile = os.path.join(currentFolder, sigleFile)
                listFile += [currentFile]
    print("Total File :{}".format(len(listFile)))
    return listFile


def EvaluateMyOpenMax(SP_Result, outputFile, labellist, classDict, detailFlag=False):
    # nameImage,GT,prob_sm,predict_sm,prob_om,predict_om
    with open("Lecog_" + outputFile, 'w') as oFile:
        # create Header File

        header_text = ["ImageName", "GT_Index", "Prob_Of_Softmax", "SoftMaxPredict",
                       "eak_1", "eak_2", "eak_3", "eak_4", "eak_5",
                       "eak_6", "eak_7", "eak_8", "eak_9", "eak_10",
                       "pow3Ak_1", "pow3Ak_2", "pow3Ak_3", "pow3Ak_4", "pow3Ak_5",
                       "pow3Ak_6", "pow3Ak_7", "pow3Ak_8", "pow3Ak_9", "pow3Ak_10",
                       "abs_eak_1", "abs_eak_2", "abs_eak_3", "abs_eak_4", "abs_eak_5",
                       "abs_eak_6", "abs_eak_7", "abs_eak_8", "abs_eak_9", "abs_eak_10",
                       "abs_pow3Ak_1", "abs_pow3Ak_2", "abs_pow3Ak_3", "abs_pow3Ak_4", "abs_pow3Ak_5",
                       "abs_pow3Ak_6", "abs_pow3Ak_7", "abs_pow3Ak_8", "abs_pow3Ak_9", "abs_pow3Ak_10"
                       ]

        Header_text = ",".join(header_text)

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

            # shape fc8_10 -> [10,1000] reduce with average to [1000]

            # shape fc8_10 -> [10,1000] reduct to power3_AK ->[10]

            abs_F8 = np.abs(fc8_10)

            pow3_AK = np.sum(np.power(fc8_10, 3), axis=1)  # find power 3 of sum product
            pow3_AK = [str(i) for i in pow3_AK]
            eak = np.sum(np.exp(fc8_10), axis=1)  # find eak of sum product
            eak = [str(i) for i in eak]


            text_power3ak = ",".join(pow3_AK)
            text_eak = ",".join(eak)


            abs_pow3_AK = np.sum(np.power(abs_F8, 3), axis=1)  # find power 3 of sum product
            abs_pow3_AK = [str(i) for i in abs_pow3_AK]
            abs_eak = np.sum(np.exp(abs_F8), axis=1)  # find eak of sum product
            abs_eak = [str(i) for i in abs_eak]


            text_abs_pwe = ",".join(abs_pow3_AK)
            text_abs_eak = ",".join(abs_eak)


            predict_sm = np.argmax(scores)  # get class index max
            prob_sm = scores[predict_sm]  # get prob of prediction

            Result_text = ",".join([ImageName, str(GTClassIndex),
                                    str(prob_sm), str(predict_sm)])
            print(ImageName)

            Result_all = "{},{},{},{},{}".format(Result_text,text_eak,text_power3ak,text_abs_eak,text_abs_pwe)


            oFile.write(Result_all + '\n')


# ---------------------------------------------------------------------------------
def main(args):
    synsetfname = args.synsetfname
    outfile = args.outfile
    featureData = args.featureData
    mode = args.Mode
    labellist = getlabellist(synsetfname)
    classDictmap = classDictMap(synsetfname)

    listFile = TravelFolder(featureData, mode)

    EvaluateMyOpenMax(listFile, outfile, labellist, classDictmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website default (<synset_words_caffe_ILSVRC12.txt>)"
    )
    parser.add_argument("--outfile", default="MyOpenresultAll.txt", help="That is a threshole default 0")
    parser.add_argument(
        "--featureData",
        default='./FeatureCloseSet/',
        help="Soft_OpenMax result default <./FeatureCloseSet/>"
    )
    parser.add_argument("--Mode", default="CloseSet",
                        help="Mode CloseSet Openset FoolingImage default <CloseSet>")

    args = parser.parse_args()
    main(args)
