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

def compute_mean_vector(category_list,featureset, labellist, layer = 'fc8'):
    for category_name in category_list:
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
        if not os.path.exists("./mean/"):
            os.mkdir("mean")
        savemat('./mean/%s.mat' %category_name, {'%s'%category_name: channel_mean_vec})

def multiproc_compute_mean_vector(params):
    return compute_mean_vector(*params)


def main(args):
    usingThread = args.thread
    category_name = args.category
    featureset = args.featureset
    labellist = getlabellist('../synset_words_caffe_ILSVRC12.txt')
    if(args.all):
        dirs = os.listdir(featureset)
        if usingThread:
            for idx in range(10):
                selected = [dirs[i] for i in range(idx,1000,10)]
                p = mp.Process(target=compute_mean_vector, args=(selected,featureset, labellist,))
                p.start()
                print("Finished {}".format(selected))
        else:
                st = time.time()
                compute_mean_vector(dirs,featureset, labellist)
                print("Total time %s secs" %(time.time() - st))
    else:
        st = time.time()
        compute_mean_vector([category_name], featureset, labellist)
        print("Total time %s secs" % (time.time() - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--featureset",
        help="Path to save matFile.",
        default="../FeatureData/",required=True
    )
    parser.add_argument("--category",
                        help="usage: python MAV_Compute.py <synset_id (e.g. n01440764)>",
                        default="n01440764",required=True)

    parser.add_argument("--all",
                        help="get all class in path",
                        action='store_true')
    parser.add_argument("--thread",
                        help="get all class in path",
                        action='store_false')
    args = parser.parse_args()
    main(args)

