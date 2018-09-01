
import os
import cv2
import numpy as np
import random
import argparse

def readlistSelectedClass(classSelected):
    listSelected = []
    with open(classSelected,'r') as clsF:
        line = clsF.readline()
        while line:
            listSelected += line.splitlines()
            line = clsF.readline()
    return listSelected

def travelInpath(opensetFile,opensetPath,classseleted):
    liseSelected = readlistSelectedClass(classseleted)

    with open(opensetFile,"w") as of :
        for f in liseSelected :
            currentpath = os.path.join(opensetPath, f)

            path, foldername = os.path.split(currentpath)
            print("Class name {}".format(foldername))

            for ff in os.listdir(currentpath):
                full_path  = os.path.join(currentpath, ff)
                datasetdetail = ",".join([full_path,str(1001)])
                of.write(datasetdetail)
                of.write("\n")


if __name__ == "__main__":
    #for example TravelInOpenset.py --opensetPath ../OpensetData --opensetFile openset_detail.txt --classSelect compareFile.txt


    paser = argparse.ArgumentParser()
    paser.add_argument("--opensetPath",
                       default="../Dataset",
                       help="select Dataset path")
    paser.add_argument("--opensetFile",
                       default="Openset_detail.txt",
                       help="Give the detail of openset file name")
    paser.add_argument("--classSelect",
                       default="compareFile.txt",
                       help="Give the compareration of class file ")
    args = paser.parse_args()


    opensetFile = args.opensetFile
    opensetPath = args.opensetPath
    classselected = args.classSelect
    travelInpath(opensetFile,opensetPath,classselected)

    print("Finished Preprocess Splited Training and Validation Dataset")
