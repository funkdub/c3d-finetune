from keras.layers.convolutional import Convolution3D
from keras.layers import Conv3D
from keras.layers.convolutional import MaxPooling3D, ZeroPadding3D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.regularizers import l2
import keras

import glob
import os
import argparse
import cv2
import numpy as np

from c3d_datagenerator import DataGenerator

os.environ["CUDA_VISIBLE_DEVICES"]="3"

batch_size = 32


source = "/datahdd/Dataset/Moments_in_Time_Mini/processed"
format = "jpg"
txt_category = "/datahdd/Dataset/Moments_in_Time_Mini/moments_categories.txt"

csv_training = "/datahdd/Dataset/Moments_in_Time_Mini/trainingSet.csv"
csv_validation = "/datahdd/Dataset/Moments_in_Time_Mini/validationSet.csv"

vid_list = glob.glob(source+'/*/*/*.'+format)
num_vid = len(vid_list)

category_name = np.loadtxt(txt_category, delimiter=',', usecols=0, dtype=str, encoding='utf-8')

category_encoding = np.loadtxt(txt_category, delimiter=',', usecols=1, dtype=None, encoding='utf-8')

print('The number of training video is : '+str(num_vid))
print('The number of categories is : '+str(len(category_encoding)))

# Load information from csv (the file is supported by data provider)
training_filename = np.genfromtxt(csv_training, delimiter=',', usecols=0, dtype=None, encoding='utf-8')
training_label = np.genfromtxt(csv_training, delimiter=',', usecols=1, dtype=None, encoding='utf-8')

validation_filename = np.genfromtxt(csv_validation, delimiter=',', usecols=0, dtype=None, encoding='utf-8')
validation_label = np.genfromtxt(csv_validation, delimiter=',', usecols=1, dtype=None, encoding='utf-8')


model = Sequential()

#model.add(BatchNormalization())
model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv1',
                        subsample=(1, 1, 1),
                        input_shape=(3, 16, 128, 128), data_format="channels_first" ))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       border_mode='valid', name='pool1' , data_format="channels_first" ))
# 2nd layer group
#model.add(BatchNormalization())
model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv2',
                        subsample=(1, 1, 1), data_format="channels_first" ))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool2', data_format="channels_first" ))
# 3rd layer group
#model.add(BatchNormalization())
model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3a',
                        subsample=(1, 1, 1), data_format="channels_first" ))
#model.add(BatchNormalization())
model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3b',
                        subsample=(1, 1, 1), data_format="channels_first" ))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool3', data_format="channels_first" ))
# 4th layer group
#model.add(BatchNormalization())
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv4a',
                        subsample=(1, 1, 1), data_format="channels_first" ))
#model.add(BatchNormalization())
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv4b',
                        subsample=(1, 1, 1), data_format="channels_first" ))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool4', data_format="channels_first" ))
# 5th layer group
#model.add(BatchNormalization())
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv5a',
                        subsample=(1, 1, 1), data_format="channels_first" ))
#model.add(BatchNormalization())
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv5b',
                        subsample=(1, 1, 1), data_format="channels_first" ))
model.add(ZeroPadding3D(padding=(0, 1, 1), data_format="channels_first" ))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                       border_mode='valid', name='pool5', data_format="channels_first" ))
model.add(Flatten())
# FC layers group
model.add(Dense(4096, activation='relu', name='fc6'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu', name='fc7'))
model.add(Dropout(.5))
model.add(Dense(487, activation='softmax', name='fc8'))

model.summary()

model.load_weights('./models/sports1M_weights.h5', 'r')
print('pre-trained model loaded')

for i, layer in enumerate(model.layers):
    print(i, layer.name)

model.layers.pop()
#model.layers.pop()
#model.layers.pop()



#model.add(Dense(4096, activation='relu', name='fc7-1'))
#model.add(Dropout(.5))
model.add(Dense(200, W_regularizer=l2(0.01),activation="linear"))

for i, layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers[:18]:
    layer.trainable = False

for layer in model.layers[18:]:
    layer.trainable = True

model.summary()

#adam = optimizers.Adam(lr=0.0001)
sgd = optimizers.SGD(lr=.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy', 'top_k_categorical_accuracy'])

filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose = 0, save_best_only = True, period = 1)
model.save('./3dConv'+'batchsize_'+str(batch_size)+'EPOCH-50-FINETUNE')

training_generator = DataGenerator(training_filename, training_label)
validation_generator = DataGenerator(validation_filename, validation_label)


model.fit_generator(generator=training_generator, epochs = 10, verbose = 1, validation_data=validation_generator)
