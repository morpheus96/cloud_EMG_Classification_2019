import os
os.environ['KERAS_BACKEND']='theano'
import sys
import keras
import numpy as np

train_data_X = np.load('train_data_X.npy')
train_data_Y = np.load('train_data_Y.npy')
test_data_X = np.load('test_data_X.npy')
test_data_Y = np.load('test_data_Y.npy')

knn_to_train=keras.models.load_model('/architecture.h5')
os.remove('/architecture.h5')
print("Successful image build!")

knn_to_train.fit(train_data_X, train_data_Y, batch_size=500, validation_split=0.8, epochs=1, verbose=0)

knn_to_train.save('/trained.h5')