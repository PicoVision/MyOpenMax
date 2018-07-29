
from tensorflow.data import Dataset, Iterator
import os
import cv2
import tensorflow as tf
#
# let's assume we have two classes in our dataset
NUM_CLASSES = 2
batch_size = 4


def input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, NUM_CLASSES)

    # read the img from file
    img_file = tf.read_file(img_path)
    # you may using opencv library
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot

def travelInpath(pathTravel,textfile):
    print("xx")
    all_file = []
    with open(textfile,"w") as filetrain:

        for f in os.listdir(pathTravel):
            currentpath = os.path.join(pathTravel, f)
            path, filename = os.path.split(currentpath)
            print("Class name {}".format(filename))
            for ff in os.listdir(currentpath):
                full_path  = os.path.join(currentpath, ff)
                all_file.append([filename,ff,full_path])
                traindetail = ",".join([full_path,filename])
                filetrain.write(traindetail)
                filetrain.write("\n")
    #for debug file name output
    # print(all_file)

    return (all_file)

def extractPahtAndClass(listFile):
    # listFile[0] => className
    # listFile[1] = > FileName
    # listFile[2] = > FilePaht
    trainImage = []
    trainLabel = []
    for eachData in listFile:
        trainImage.append(eachData[2])
        trainLabel.append(int(eachData[0]))

    return (trainImage,trainLabel)


if __name__ == "__main__":

    listFile = travelInpath("./Dataset/train/","traindetail.txt")

    #Dataset path
    train_imgs,train_labels = extractPahtAndClass(listFile)
    print(train_imgs)
    print(train_labels)

    train_imgs = tf.constant(train_imgs)
    train_labels = tf.constant(train_labels)

    # create TensorFlow Dataset objects
    tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))

    tr_data = tr_data.map(input_parser).batch(batch_size)
    # tr_data = tr_data.map(input_parser, num_parallel_calls=8).batch(batch_size).\
    #     shuffle(batch_size)

    # create TensorFlow Iterator object
    iterator = Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
    next_element = iterator.get_next()

    # initialization the datasets operation
    training_init_op = iterator.make_initializer(tr_data)

    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(training_init_op)

        # get each element of the training dataset until the end is reached
        while True:
            try:
                elem = sess.run(next_element)
                print("-------------------")
                for imageinBat in elem[1]:
                    print(imageinBat)
                    print("xxx")
            except tf.errors.OutOfRangeError:
                print("End of training dataset. Reset and Loop again")
                sess.run(training_init_op)
