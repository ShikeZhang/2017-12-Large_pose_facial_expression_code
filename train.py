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


# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_file, image_shape=(250, 250), mode='file', 
			output_path='your_data_path.h5', categorical_labels=False, normalize=True)

# Load HDF5 dataset
import h5py
h5f = h5py.File('your_data_path.h5', 'r')  
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
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 2, activation='sigmoid')


network = tflearn.regression(network, optimizer='Momentum', learning_rate=0.01,
                     loss='categorical_crossentropy', metric='accuracy', to_one_hot = True, n_classes=2)


model = tflearn.DNN(network, tensorboard_verbose=3)

model.fit(X, Y, n_epoch=50, validation_set=0.5, batch_size = 256, 
			run_id="cg_network", show_metric=True, shuffle=True)



model.save('your_data_path.tfl')



