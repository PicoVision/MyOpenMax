# -*- coding: utf-8 -*-

import scipy as sp
import sys
import os, glob
import os.path as path
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat
import argparse
import multiprocessing as mp

#------------------------------------------------------------------------------------------
def getlabellist(fname):
    """ Read synset file for ILSVRC 2012
    """

    imagenetlabels = open(fname, 'r').readlines()
    labellist  = [i.split(' ')[0] for i in imagenetlabels]        
    return labellist

#------------------------------------------------------------------------------------------
def compute_channel_distances(mean_train_channel_vector, features, category_name):
    """
    Input:
    ---------
    mean_train_channel_vector : mean activation vector for a given class. 
                                It can be computed using MAV_Compute.py file
    features: features for the category under consideration
    category_name: synset_id

    Output:
    ---------
    channel_distances: dict of distance distribution from MAV for each channel. 
    distances considered are eucos, cosine and euclidean
    """

    eucos_dist, eu_dist, cos_dist = [], [], []
    for channel in range(features[0].shape[0]):
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        # compute channel specific distances
        for feat in features:
            eu_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])]
            cos_channel += [spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
            eu_cos_channel += [spd.euclidean(mean_train_channel_vector[channel, :], feat[channel, :])/200. +
                               spd.cosine(mean_train_channel_vector[channel, :], feat[channel, :])]
        eu_dist += [eu_channel]
        cos_dist += [cos_channel]
        eucos_dist += [eu_cos_channel]

    # convert all arrays as scipy arrays
    eucos_dist = sp.asarray(eucos_dist)
    eu_dist = sp.asarray(eu_dist)
    cos_dist = sp.asarray(cos_dist)

    # assertions for length check
    assert eucos_dist.shape[0] == 10
    assert eu_dist.shape[0] == 10
    assert cos_dist.shape[0] == 10
    assert eucos_dist.shape[1] == len(features)
    assert eu_dist.shape[1] == len(features)
    assert cos_dist.shape[1] == len(features)

    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
    return channel_distances
    
#------------------------------------------------------------------------------------------
def compute_distances(meanPath, categoryList,
                      featurefilepath_list, layer = 'fc8'):
    """
    Input:
    -------
    mav_fname : path to filename that contains mean activation vector
    labellist : list of labels from ilsvrc 2012
    category_name : synset_id

    """
    for category_name in categoryList:
        mav_fname = "{}/{}.mat".format(meanPath, category_name)
        featurefilepath = "{}/{}".format(featurefilepath_list,category_name)
        labellist = getlabellist('../synset_words_caffe_ILSVRC12.txt')



        mean_feature_vec = loadmat(mav_fname)[category_name]
        # print('%s/%s/*.mat' %(featurefilepath, category_name))
        featurefile_list = glob.glob('%s/*.mat' %featurefilepath)

        correct_features = []
        for featurefile in featurefile_list:
            # print(idx)
            # if idx > 10 :
            #     break
            try:
                img_arr = loadmat(featurefile)
                predicted_category = labellist[img_arr['scores'].argmax()]
                if predicted_category == category_name:
                    correct_features += [img_arr[layer]]
            except TypeError:
                continue

        distance_distribution = compute_channel_distances(mean_feature_vec, correct_features, category_name)
        if not os.path.exists("./distance/"):
            os.mkdir("distance")
        savemat('./distance/{}_distances.mat'.format(category_name), distance_distribution)

#------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--meanPath",
        help="Path to save matFile.",
        default="./mean/"
    )
    parser.add_argument(
        "--featurePath",
        help="Feature path with same class of mean File",
        default="./featurePath/"
    )
    parser.add_argument("--category",
                        help="usage: python MAV_Compute.py <synset_id (e.g. n01440764)>",
                        default="n01440764")

    parser.add_argument("--all",
                        help="calculate all category name in feature path",
                        action="store_true")
    args = parser.parse_args()

    meanPath = args.meanPath
    feature_path = args.featurePath

    if args.all :
        listcategory = os.listdir(args.featurePath)
        for idx in range(30):
            selected = [listcategory[i] for i in range(idx, 1000, 30)]
            # compute_distances(meanPath,feature_path, selected)
            p = mp.Process(target=compute_distances, args=(meanPath,selected,feature_path, ))
            p.start()
            print("Finished {}".format(selected))

    else :
        listcategory = os.listdir(args.featurePath)
        compute_distances(meanPath,  listcategory,feature_path)

if __name__ == "__main__":
    #print("python compute_distances.py n01440764 ../data/mean_files/n01440764.mat ../data/train_features/n01440764/")
    main()
