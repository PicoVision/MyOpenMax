import numpy as np
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input


def createModel(inputShape=(227,227,3),class_out = 1000):
    input_images = Input(shape=inputShape)
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same')(input_images)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (4, 4), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(class_out, kernel_initializer=init, activation='softmax')(x)

    model = Model(inputs=images, outputs=outputs)
    return model