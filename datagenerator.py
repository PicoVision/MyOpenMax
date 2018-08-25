# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, mode, batch_size, num_classes, class_file, txt_file="", img_paths="",labels="",shuffle=False,
                 buffer_size=20):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.num_classes = num_classes

        # convert name class to label
        self.class_file = class_file
        self._classMaping()

        # retrieve the data from the text file


        self.txt_file = txt_file
        if(self.txt_file != ""):
            self._read_txt_file()
        else:
            # assert(img_paths=="" or labels ==""," img_pahts or labels is null")
            self._setDataset(img_paths=img_paths,labels=labels)

        #remap label class
        self._remapToLabel()

        # print(self.labels)

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        # dataset_source = (tf.data.Dataset.from_tensor_slices(input_tensors).
        #                   map(pre_processing_func, num_parallel_calls=NUM_THREADS).
        #                   prefetch(BUFFER_SIZE).
        #                   flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x)).
        #                   shuffle(BUFFER_SIZE))  # my addition, probably necessary though
        # fixed from https://stackoverflow.com/questions/47411383/parallel-threads-with-tensorflow-dataset-api-and-flat-map


        if mode == 'training':
            # data = data.map(map_func=self._parse_function_train(), num_parallel_calls=8)
            # data = data.batch(batch_size=buffer_size)
            data = data.apply(tf.contrib.data.map_and_batch(
                map_func=self._parse_function_train,batch_size=batch_size,num_parallel_batches=4))
            data = data.prefetch(buffer_size=buffer_size)
        elif mode == "validation":
            data = data.apply(tf.contrib.data.map_and_batch(
                map_func=self._parse_function_inference, batch_size=batch_size, num_parallel_batches=4))
            data = data.prefetch(buffer_size=buffer_size)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        self.data = data

    def _remapToLabel(self):
        labelInt = []
        for idx in range(len(self.labels)):
            labelInt.append(int(self.class_map[self.labels[idx]]))
        self.labels = labelInt

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(',')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def _setDataset(self,img_paths,labels):
        self.img_paths = img_paths
        self.labels = labels

    def _classMaping(self):
        self.class_map  = {}
        with open(self.class_file, 'r') as cf:
            line = cf.readline()
            while line:
                items = line.split(',')
                # print(items)
                self.class_map[items[0]] = str(items[1]).strip("\n")
                line = cf.readline()
        # print(self.class_map)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)[:, :, ::-1]

        return img_centered, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)[:, :, ::-1]

        return img_centered, one_hot
