import cv2
import numpy as np
import tensorflow as tf
import os
from scipy.io import savemat
import argparse

def oversample(images, crop_dims):
    """
    Crop images into the four corners, center, and their mirrored versions.
    Parameters
    ----------
    image : iterable of (H x W x K) ndarrays
    crop_dims : (height, width) tuple for the crops.
    Returns
    -------
    crops : (10*N x H x W x K) ndarray of crops for number of inputs N.
    """
    # Dimensions and center.
    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]  # flip for mirrors
    return crops


def getClassMapDict(ClassFile):
    classDict = {}
    with open(ClassFile, 'r') as cF:
        line = cF.readline()
        while line:
            classname = line.split(',')
            classDict[classname[1]] = classname[0]
            line = cF.readline()
    return classDict

def read_train_detail(detailFile,shuffle=False):
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

    return list(zip(img_paths, labels))




def main(args):
    savePath_mat = args.SavePath
    weightFile = args.weightFile
    TrainingFile = args.Filedetail
    ClassMapFile =args.classmap

    tf.reset_default_graph()

    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    # Load Model
    if not os.path.exists("{:s}.meta".format(weightFile)):
        print("Create Normal AlexNet Model!!")
        assert False, "Can't found Alexnet Graph File"
    else:
        print("Alexnet Loading now!! ")
        model = tf.train.import_meta_graph('{:s}.meta'.format(weightFile))

    with tf.Session() as sess:
        # Initialize all variables

        sess.run(tf.global_variables_initializer())
        tensorA = [tensor.name for tensor in tf.get_default_graph().get_operations()]
        # print(tensorA)
        # if os.path.exists('{:s}'.format(weightFile)):
        print("Loading... Weight...")
        model.restore(sess, weightFile)

        x_input  = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("Placeholder_2:0")
        f8= tf.get_default_graph().get_tensor_by_name("fc8/fc8:0")
        f7 = tf.get_default_graph().get_tensor_by_name("fc7/fc7:0")
        prob = tf.nn.softmax(f8)
        score = tf.reduce_mean(prob,axis=0)


        TrainFiles = read_train_detail(TrainingFile)
        for trainFile in TrainFiles:
            print("{:s} Class {:s}".format(trainFile[0],trainFile[1]))
            imageFile = trainFile[0]
            className = trainFile[1]
            #### Create Directory File
            if not os.path.exists(savePath_mat):
                os.mkdir(savePath_mat)
            if not os.path.exists("{:s}/{:s}".format(savePath_mat,className)):
                os.mkdir("{:s}/{:s}".format(savePath_mat,className))

            fileNameFeature = "{:s}/{:s}/{:s}.mat".format(savePath_mat,className,os.path.basename(imageFile).split(".")[0])
            # print(fileNameFeature)
            # print("------Predict with Open CV--------")
            img = cv2.imread(imageFile)
            img = cv2.resize(img.astype(np.float32),(256,256))
            img = img - imagenet_mean
            img = img.reshape(-1,256,256,3)
            img_oversampling = oversample(img, (227, 227))

            # Run the session and calculate the class probability
            # probs = sess.run(softmax, feed_dict={x: img_oversampling, keep_prob: 1})

            feature_dict = {}
            feature_dict['IMG_NAME'] = "{:s}/{:s}/{:s}".format(savePath_mat,className,os.path.basename(imageFile))
            pFc7,pFc8,pProb,pScore = sess.run([f7,f8,prob,score], feed_dict={x_input: img_oversampling, keep_prob: 1})
            feature_dict['fc7'] = np.asarray(pFc7)
            feature_dict['fc8'] = np.asarray(pFc8)
            feature_dict['prob'] = np.asarray(pProb)
            feature_dict['scores'] = np.asarray(pScore)
            savemat(fileNameFeature, feature_dict)


if __name__ == "__main__":

    #for example ExtractFeature.py --SavePath ../FeatureData/ --weightFile ../BestWeight/weight
    #--Filedetail ../TrainingFile_detail.txt  --classmap ../Classmap.txt

    #for example ExtractFeature.py --SavePath ../FeatureHackImage --weightFile ../BestWeight/weight --Filedetail ../Hacked_detail.txt --classmap ../HackImageMap.txt
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--SavePath",
        help="Path to save matFile.",
        default="./FeatureData/"
    )

    parser.add_argument("--weightFile",
                        help="weight File",
                        default="../BestWeight/weight")

    parser.add_argument("--Filedetail",
                        help="File detail that have path and label of image",
                        default="../TrainingFile_detail.txt")

    parser.add_argument("--classmap",
                        help="FileClass mapping",
                        default="../Classmap.txt")

    args = parser.parse_args()
    # weightFile = "../BestWeight/weight"
    # TrainingFile = "../TrainingFile_detail.txt"
    # ClassMapFile = "../Classmap.txt"

    main(args)
