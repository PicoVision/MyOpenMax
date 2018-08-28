

import os, sys
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pickle
import os.path as path
import multiprocessing as mp
import numpy as np
import argparse



def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def getlabellist(fname):

    imagenetlabels = open(fname, 'r').readlines()
    labellist  = [i.split(' ')[0] for i in imagenetlabels]        
    return labellist

def compute_mean_vector(category_name,featureset, labellist, layer = 'fc8'):
    print(category_name)
    featurefile_list = glob.glob('{}/{}/*.mat'.format(featureset, category_name))
    
    # gather all the training samples for which predicted category
    # was the category under consideration
    correct_features = []
    for featurefile in featurefile_list:
        try:
            img_arr = loadmat(featurefile)
            predicted_category = labellist[img_arr['scores'].argmax()]
            #bee debug
            array_prob = img_arr['prob']
            array_f8 = img_arr['fc8']
            array_score = img_arr['scores']
            a_prob = sp.mean(array_prob,axis=0)
            fc8_mean = sp.mean(array_f8,axis=0)
            fc8_softmax_mean = softmax(fc8_mean)
            #End Be debug
            if predicted_category == category_name:
                correct_features += [img_arr[layer]]
        except TypeError:
            continue
    
    # Now compute channel wise mean vector
    channel_mean_vec = []
    for channelid in range(correct_features[0].shape[0]):
        channel = []
        for feature in correct_features:
            channel += [feature[channelid, :]]
        channel = sp.asarray(channel)
        assert len(correct_features) == channel.shape[0]
        # Gather mean over each channel, to get mean channel vector
        channel_mean_vec += [sp.mean(channel, axis=0)]

    # this vector contains mean computed over correct classifications
    # for each channel separately
    channel_mean_vec = sp.asarray(channel_mean_vec)
    savemat('%s.mat' %category_name, {'%s'%category_name: channel_mean_vec})

def multiproc_compute_mean_vector(params):
    return compute_mean_vector(*params)

def main(args):

    category_name = args.category
    featureset = args.featureset
    st = time.time()
    labellist = getlabellist('../synset_words_caffe_ILSVRC12.txt')
    compute_mean_vector(category_name,featureset, labellist)
    print("Total time %s secs" %(time.time() - st))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--featureset",
        help="Path to save matFile.",
        default="../FeatureData/"
    )
    parser.add_argument("--category",
                        help="usage: python MAV_Compute.py <synset_id (e.g. n01440764)>",
                        default="n01440764")

    args = parser.parse_args()
    main(args)

