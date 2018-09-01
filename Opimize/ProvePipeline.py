import tensorflow as tf
import numpy as np
NUM_EPOCHS = 10

def mapFunction():
    a = 0

dataset = tf.data.TFRecordDataset("train.tfrecords",num_parallel_reads=8)
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(NUM_EPOCHS)
dataset = dataset.map(map_func=mapFunction,num_parallel_calls=16)
dataset = dataset.batch(50)
dataset = dataset.prefetch(2)
with tf.Session() as session:
    session
    print(a)