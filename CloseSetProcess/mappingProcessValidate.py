import numpy as np
import os
import argparse


def getlabellist(synsetfname):
    """ read sysnset file as python list. Index corresponds to the output that
    caffe provides
    """

    categorylist = open(synsetfname, 'r').readlines()
    labellist = [category.split(' ')[0] for category in categorylist]
    return labellist


def extractLabelOfGDTH(gdThFile):
    lableList = []
    with open(gdThFile,'r') as gthF:
        line = gthF.readline()
        while line:
            line = line.splitlines()
            line = line[0].split(" ")[1]
            lableList += [line]
            line = gthF.readline()
    return lableList


def main(args):
    ImagePath = args.PathImage  # expect all file in Folder
    gdThFile = args.goundthFile
    outfile = args.outfile
    classMapFile = args.synsetfname


    if not os.path.exists(gdThFile) or not os.path.exists(ImagePath) :
        exit()
    else:
        labelList = extractLabelOfGDTH(gdThFile)
        listClass = getlabellist(classMapFile)
        # print(labelList)
        # print(dictClass)
        with open(outfile,'w') as outF:
            for idx, eachImageFile in  enumerate(os.listdir(ImagePath)):
                if idx < 50000:
                    label = int(labelList[idx])
                    ImagepathMap = os.path.join(ImagePath, eachImageFile)
                    mappintString = ",".join([ImagepathMap,listClass[label]])
                    print(mappintString)
                    outF.write(mappintString+"\n")

if __name__ == '__main__':
    #create closeset_detail
    paser = argparse.ArgumentParser()
    paser.add_argument("--PathImage",help="image path")
    paser.add_argument("--goundthFile",help="ground touth file")
    paser.add_argument("--outfile",default="closeset_detail.txt")
    paser.add_argument(
        "--synsetfname",
        default='../synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website"
    )
    args = paser.parse_args()

    main(args)
