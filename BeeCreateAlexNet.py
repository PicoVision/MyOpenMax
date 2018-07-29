import os

import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator

from tensorflow.data import Iterator

# File Path desc
train_file = "traindetail.txt"
val_file = "valdetail.txt"

# Network params
learning_rate = 0.001
dropout_rate = 0.5
num_classes = 2
batch_size = 2
num_epochs = 1000000
display_step = 0  #check acc per epoach
train_layers = ['fc8', 'fc7', 'fc6']


with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    # val_data = ImageDataGenerator(val_file,
    #                               mode='inference',
    #                               batch_size=batch_size,
    #                               num_classes=num_classes,
    #                               shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

#intial trainning set
training_init_op = iterator.make_initializer(tr_data.data)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers,weights_path="./weights/bvlc_alexnet.npy")


# Link variable to model output
score = model.fc8


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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)


# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="ACC")


# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter("./Graph")

with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    print("Weight Loading...")
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))


    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)
        print(range(train_batches_per_epoch))
        for step in range(train_batches_per_epoch):
            print("write")
            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})

            writer.add_summary(s, epoch * train_batches_per_epoch + step)

            # if step % display_step == 0:
            #     print("writeed")



