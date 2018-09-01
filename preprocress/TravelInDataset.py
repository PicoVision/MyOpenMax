
import os
import cv2
import numpy as np
import random
import argparse
def travelInpath(pathTravel,trainingfile,validationfile,classMapFile,numsplit,shuffle = False):

    classFile = {}
    with open(trainingfile,"w") as tnFile ,\
            open(validationfile,"w")  as valFile,\
            open(classMapFile,"w") as clFile:


        for idxclass,f in enumerate(os.listdir(pathTravel)):
            currentpath = os.path.join(pathTravel, f)
            path, foldername = os.path.split(currentpath)
            print("Class name {}".format(foldername))
            clFile.write(",".join([foldername,str(idxclass)]))
            clFile.write("\n")

            all_file_infolder = []
            for ff in os.listdir(currentpath):
                full_path  = os.path.join(currentpath, ff)
                datasetdetail = ",".join([full_path,foldername])
                all_file_infolder.append(datasetdetail)
            if shuffle:
                random.shuffle(all_file_infolder)
            if numsplit > 0:
                traindetail =all_file_infolder[:-numsplit]
                valdetail = all_file_infolder[-numsplit:]
            else:
                traindetail = all_file_infolder
                valdetail = []
            for i in traindetail:
                tnFile.write(i)
                tnFile.write("\n")

            for i in valdetail:
                valFile.write(i)
                valFile.write("\n")



def read_train_detail(detailFile,shuffle=True):
    """Read the content of the text file and store it into lists."""
    img_paths = []
    labels = []
    with open(detailFile, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(',')
            img_paths.append(items[0])
            labels.append(items[1].strip("\n"))
            line = f.readline()
    if shuffle:
        c = list(zip(img_paths, labels))
        random.shuffle(c)

        img_paths, labels = zip(*c)
        #print(labels)

    return img_paths,labels

if __name__ == "__main__":
    #for example TravelInDaset.py --datasetPath ../HackedImage --trainingfile Hacked_detail.txt
    # for example TravelInDaset.py --datasetPath F:/LargeDataset/2012/Data
    #  --trainingfile TrainingFile_detail.txt  --validationfile ValidationFile_detail.txt
    # --spiltnum 50

    paser = argparse.ArgumentParser()
    paser.add_argument("--datasetPath",
                       default="../Dataset",
                       help="select Dataset path")
    paser.add_argument("--datasetfile",default="AllDataset_Detail.txt",
                       help="Give the dataset file name")
    paser.add_argument("--trainingfile",
                       default="TrainingFile_detail.txt",
                       help="Give the training file name")
    paser.add_argument("--validationfile",
                       default="ValidationFile_detail.txt",
                       help="Give the Validation file name")
    paser.add_argument("--spiltnum"                      ,
                       default=-1,
                       help="Give the total number that split from trining to validation per class")
    args = paser.parse_args()

    ClassmapFile = "./Classmap.txt"
    AllDataset = args.datasetfile
    TrainingFile = args.trainingfile
    ValidationFile = args.validationfile
    DatasetPath = args.datasetPath
    numsplit = args.spiltnum
    travelInpath(DatasetPath,TrainingFile,ValidationFile,ClassmapFile,numsplit)

    print("Finished Preprocess Splited Training and Validation Dataset")
