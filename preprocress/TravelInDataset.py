
import os
import cv2
import numpy as np
import random

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

            traindetail =all_file_infolder[:-numsplit]
            valdetail = all_file_infolder[-numsplit:]

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

    AllDataset = "../AllDataset_Detail.txt"
    TrainingFile = "../TrainingFile_detail.txt"
    ValidationFile = "../ValidationFile_detail.txt"
    ClassmapFile = "../Classmap.txt"
    DatasetPath = "../Dataset/"
    numsplit = 5 #perclass
    travelInpath(DatasetPath,TrainingFile,ValidationFile,ClassmapFile,numsplit)

    print("Finished Preprocess Splited Training and Validation Dataset")
