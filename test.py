# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
tf.reset_default_graph()
from tflearn.data_utils import load_csv

csv.field_size_limit(sys.maxsize)

# Load path/class_id image file:
#dataset_file ="your file path"

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(250, 250), mode='file', output_path='/Users/coco/Desktop/CG_data/dataset_test.h5', categorical_labels=False, normalize=True)

# Load HDF5 dataset
import h5py
h5f = h5py.File('your file path.h5', 'r')
X = h5f['X']
Y = h5f['Y']


network = tflearn.input_data(shape=[None, 250, 250, 3])
network = tflearn.conv_2d(network, 64, 10, 4,activation='relu', regularizer="L1")
network = tflearn.max_pool_2d(network, 2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 128, 11, activation='relu', regularizer="L1")
network = tflearn.max_pool_2d(network, 2)
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 128, activation='relu')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 32, activation='relu')
network = tflearn.fully_connected(network, 2, activation='sigmoid')

#top1 = tflearn.metrics.Top_k(k=1)

network = tflearn.regression(network, optimizer='Momentum', learning_rate=0.01,
                     loss='categorical_crossentropy', metric='accuracy', to_one_hot = True, n_classes=2)


model = tflearn.DNN(network, tensorboard_verbose=0)

model.load('/Users/coco/Desktop/CG_data/model.tfl')

score = model.evaluate(X, Y)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))


# Run the model on one example
prediction = model.predict([X[0]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[1]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[2]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[3]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[4]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[5]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[6]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[7]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[8]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[9]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[10]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[11]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[12]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[13]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[14]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[15]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[16]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[17]])
print("Prediction: %s" % str(prediction[0]))
prediction = model.predict([X[18]])
print("Prediction: %s" % str(prediction[0]))




#model.save('file path')



