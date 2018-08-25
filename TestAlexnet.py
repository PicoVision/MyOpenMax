#some basic imports and setups
# This file to extract fc8 feature and save to .m type
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet


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


tf.reset_default_graph()
#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, './Dataset/n01484850')

# get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.JPEG')]

# load all images
imgs = []
for f in img_files:
    imgs.append(cv2.imread(f))


# placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

# create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000, [],weights_path="./weights/bvlc_alexnet.npy")

# define activation of last layer as score
score = model.fc8

# create op to calculate softmax
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Loop over all images
    for i, image in enumerate(imgs):
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (256, 256))
        # Subtract the ImageNet mean
        img -= imagenet_mean
        img = img.reshape(-1,256,256,3)
        img_oversampling = oversample(img,(227,227))

        # Reshape as needed to feed into model
        #img = img.reshape((1, 227, 227, 3))

        # Run the session and calculate the class probability
        probs = sess.run(softmax, feed_dict={x: img_oversampling, keep_prob: 1})

        # Get the class name of the class with the highest probability
        # class_name = class_names[np.argmax(probs)]
        avgProb = np.mean(probs,axis=0)
        print(np.argmax(avgProb))
        print(np.argmax(probs,axis=1))
        # print(class_name)

        # Plot image with class name and prob in the title
