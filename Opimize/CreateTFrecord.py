import cv2
import numpy as np
import tensorflow as tf
import sys
def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def getClassMapDict(ClassFile):
    classDict = {}
    with open(ClassFile, 'r') as cF:
        line = cF.readline()
        while line:
            classname = line.split(',')
            classDict[classname[0]] = classname[1]
            line = cF.readline()
    return classDict


def extractData(FileDetail):

    """Read the content of the text file and store it into lists."""
    img_paths = []
    labels = []
    with open(FileDetail, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(',')
            img_paths.append(items[0])
            labels.append(items[1].strip("\n"))
            line = f.readline()

    return list(zip(img_paths, labels))


TrainingFile = "../TrainingFile_detail.txt"
ClassMapFile = "../Classmap.txt"


train_addrs = extractData(TrainingFile)
classDict = getClassMapDict(ClassMapFile)

train_filename = 'train.tfrecords'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for idx,train_detail in enumerate(train_addrs):
    # print how many images are saved every 1000 images
    if not idx % 1000:
        print('Train data: {}/{}'.format(idx, len(train_addrs)))
        sys.stdout.flush()
    # Load the image
    img = load_image(train_detail[0])
    label = int(classDict[train_detail[1]])
    # Create a feature
    feature = {'label': _int64_feature(label),
               'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()