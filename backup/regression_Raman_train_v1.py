# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:24:22 2019

@author: Chuanzhen Hu
"""

from tensorflow import keras
from sklearn import preprocessing
import numpy as np
import os

# Set default decvice: GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###############################################################################
def conv_model(input_shape, output_shape, dropout_rate, learning_rate):
    # create CNN model
    model = keras.Sequential()
    
    # layer 1  [None, cols, channels]
    model.add(keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', input_shape=input_shape)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=2))
    
    # layer 2
    model.add(keras.layers.Conv1D(filters=16, kernel_size=5, padding='same')) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=2))
    
    # layer 3
    model.add(keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.AveragePooling1D(pool_size=2, strides=2))
    
    # layer 4
    model.add(keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    
    # layer 5
    model.add(keras.layers.Conv1D(filters=64, kernel_size=5, padding='same')) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    # flatten layer
    model.add(keras.layers.Flatten())
    # dropout layer
    model.add(keras.layers.Dropout(rate=dropout_rate))
    # dense layer
    model.add(keras.layers.Dense(units=output_shape))
    
    # Take a look at the model summary
    model.summary()
    
    model.compile(loss=keras.losses.mean_squared_error,
                 optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=1e-8),
                 metrics=['accuracy']) # mse
    return model
###############################################################################
def step_decay(epoch):
    lr = 1e-3
    drop_factor = 0.1
    drop_period = 20
    lrate = lr*np.math.pow(drop_factor, np.math.floor((1+epoch)/drop_period))
#    decay_rate.append(lrate)
    return lrate
###############################################################################


def main(unused_argv):
    # set super-parameters
    max_epoch = 50
    batch_size = 500
    learning_rate = 0
    dropout_rate = 0.5
    validation_ratio = 0.2

    # read in train datasets and get sizes
    print("Step 1: read in train and test data")
    X = np.load('./spectra_data/X_train.npy')
    Y = np.load('./spectra_data/y_train.npy')
    split_marker = np.int64(np.round((1-validation_ratio)*Y.shape[0]))

    # normalization: zero-mean along column, with_std=False
    X_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X[:split_marker, :])
    X[:split_marker, :] = X_scaler.transform(X[:split_marker, :])
    X[split_marker:, :] = X_scaler.transform(X[split_marker:, :])
    np.save('./RamanNet/X_scale_mean.npy', X_scaler.mean_)
    np.save('./RamanNet/X_scale_std.npy', X_scaler.scale_)
    
    # reshape train data
    X = np.reshape(X, [X.shape[0], X.shape[1], 1])
    
    # define train size 
    input_shape = X.shape[1:]
    output_shape = Y.shape[1]
    
    print("###################################################################")
    print("Step 2: create CNN model")
    if os.path.exists('./RamanNet/regression_model.h5'):
        model = keras.models.load_model('./RamanNet/regression_model.h5')
        model.summary()
        print('Load saved model and train again!!!')
    else:
        model = conv_model(input_shape=input_shape, output_shape=output_shape, 
                           dropout_rate=dropout_rate, learning_rate=learning_rate)
        print('Create new model and train!!!')
    print("###################################################################")
    print("Step 3: train CNN model")
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./RamanNet/regression_model.h5', verbose=1, save_best_only=True)
    ##################################################################################################################
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./RamanNet/logs', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X[:split_marker, :, :], Y[:split_marker, :], batch_size=batch_size, epochs=max_epoch, 
              validation_data=(X[split_marker:, :, :], Y[split_marker:, :]), callbacks=[lrate, checkpointer, tbCallBack])
    print('Finsh training!!!')
    
#     tensorboard --logdir E:/CNN/Concentration_Network/RamanNet/logs 
#     anaconda prompt
    
if __name__ == "__main__":
    main(0)