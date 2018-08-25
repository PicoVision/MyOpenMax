import os
import cv2
import numpy as np
import tensorflow as tf
import math
import random
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from tensorflow.data import Iterator
from tqdm import tqdm

def read_train_detail(train_file,shuffle=False):
    """Read the content of the text file and store it into lists."""
    img_paths = []
    labels = []
    with open(train_file, 'r') as f:
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


imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
# File Path desc
train_file = "./TrainingFile_detail.txt"
Validate_file = "./ValidationFile_detail.txt"

class_file = "./Classmap.txt"
checkpoint_path = "./ckp/"
# Network params
learning_rate = 0.00001
dropout_rate = 0.5
num_classes = 1000
batch_size = 1
num_epochs = 100
display_step = 20  #check acc per epoach
train_layers = ['fc8', 'fc7', 'fc6']


#read all image path config
train_img_paths,train_labels = read_train_detail(train_file)
validate_img_paths,validate_labels = read_train_detail(Validate_file)

print("Total Dataset {}".format(int(len(train_labels)+len(validate_labels))))
print("Split to Training {} and to Validation {}".format(len(train_labels),len(validate_labels)))

with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 class_file=class_file,
                                 shuffle=False,
                                 img_paths=train_img_paths,
                                 labels=train_labels)
    val_data = ImageDataGenerator(mode='validation',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 class_file= class_file,
                                 shuffle=False,
                                 img_paths=validate_img_paths,
                                 labels=validate_labels)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()


# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)



# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size/batch_size))



# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, [],weights_path="./weights/bvlc_alexnet.npy")


# Link variable to model output
score = model.fc8

prob_predic = tf.nn.softmax(score)
prediction=tf.argmax(prob_predic,1)
# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))


# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)



# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="ACC")


# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)
tf.summary.scalar('Acc', accuracy)


# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter("./Graph")

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()


with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    print("Weight Loading...")
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))

    best_acc = 0
    # Loop over number of epochs

    # Initialize iterator with the training dataset
    sess.run(training_init_op)

    for step in range(train_batches_per_epoch):

        # get next batch of data
        img_batch, label_batch,filename = sess.run(next_batch)


        print(filename[0].decode('ascii'))
        img2 = cv2.imread(filename[0].decode('ascii'))
        img2 = cv2.resize(img2,(227,227))
        img2 = img2-imagenet_mean
        cv2.imshow("Image opencv",img2)
        cv2.imshow("Image tensor",img_batch[0])
        print(img_batch[0]-img2)
        cv2.waitKey(0)



        # print(np.argmax(label_batch,1))
        # print(img_batch[0])
        # cv2.imshow("Image",img_batch[0].reshape(227,227,3))
        # cv2.waitKey(0)

        # And run the training op
        # sess.run(train_op, feed_dict={x: img_batch,
        #                               y: label_batch,
        #                               keep_prob: dropout_rate})

        # Generate summary with the current batch of data and write to file
        print(np.argmax(label_batch, axis=1))
        pprob = sess.run([prediction], feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1})

        pprob2 = sess.run([prediction], feed_dict={x: img2.reshape(-1,227,227,3),
                                                    y: label_batch,
                                                    keep_prob: 1})
        print(pprob)
        print(pprob2)
