'''  To run this:
     Log in to your AWS instance
     mkdir data
     pip install h5py
     While the pip install is running, from your home machine
     scp data/*.npy to your AWS instance's data directory
     Back on your AWS instance:
     launch python
     copy and paste this code
'''

# https://www.dropbox.com/s/ka65x90tdsb68bc/genre_cohorts_2x.npy
# https://www.dropbox.com/s/p3jif1u7sureydp/genre_cohorts_0x.npy
# https://www.dropbox.com/s/xpp7ab21zw34cbo/img_rgb_color_2x.npy
# https://www.dropbox.com/s/8lcqdi22lqeif05/img_rgb_color_0x.npy

import keras
from keras import backend as K
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.callbacks import LearningRateScheduler

import numpy as np

import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('white')

def do_one_genre(single_genre_index,
                 fc_size_1,
                 fc_size_2):

    data = np.load('data/img_rgb_color_0x.npy', mmap_mode='r')
    labels = np.load('data/genre_cohorts_0x.npy', mmap_mode='r')[:,single_genre_index]
    m = labels.mean()
    class_weights = {0: 1/(1-m),
                     1: 1/m,
                     } # single-label

    validation_data = np.load('data/img_rgb_color_2x.npy', mmap_mode='r')
    validation_labels = np.load('data/genre_cohorts_2x.npy', mmap_mode='r')[:,single_genre_index]

    # We want to optimize fscore. So we'll need a method
    # that can be used both as a metric (so we can report
    # on it after each epoch) and as an objective function
    # (so we can try to optimize for it)
    def fscore(y_true, y_pred):
        # y_pred = K.transpose(y_pred)
        TruePos = K.sum(K.minimum(y_pred, y_true))
        TrueNeg = K.sum(K.minimum((1-y_pred), (1-y_true)))
        FalsePos = K.sum(K.minimum(y_pred, (1-y_true)))
        FalseNeg = K.sum(K.minimum((1-y_pred), y_true))
        Precision = TruePos / (TruePos + FalsePos + 1e-10)
        Recall = TruePos / (TruePos + FalseNeg + 1e-10)
        F_Score = 2 * (Precision * Recall) / (Precision + Recall)
        return F_Score

    # We want to maximize fscore which means we want to
    # minimize -fscore
    def fscore_loss(y_true, y_pred):
        return 1-fscore(y_true, y_pred)

    '''
    ## TRY A CONV NN FROM SCRATCH
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     data_format='channels_last',
                     input_shape=(48, 48, 3),  # copied from above
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(labels.shape[1], activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',   # this is critical!
                  metrics=['accuracy'])
    model.summary()
    '''

    # CNN Based on pre-trained
    base_model = VGG16(weights='imagenet',
                       input_shape=(48, 48, 3),
                       include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Next section based on
    # https://keras.io/applications/#usage-examples-for-image-classification-models

    # Take the output features from the VGG16 model...
    x = base_model.output

    # ... add a pooling layer ... 
    x = GlobalAveragePooling2D()(x)

    # ... and now put a couple of fully-connected layers
    # on top of that ...
    x = Dense(fc_size_1, activation='relu',
              kernel_initializer='he_normal',
              #kernel_regularizer=regularizers.l2(1e-6),
              #activity_regularizer=regularizers.l1(1e-6)
    )(x)
    x = Dense(fc_size_2, activation='relu',
              kernel_initializer='he_normal',
              # kernel_regularizer=regularizers.l2(1e-6),
              activity_regularizer=regularizers.l1(1e-5)
    )(x)

    # ... and a logistic layer to extract our prediction
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    model.compile(optimizer=keras.optimizers.Adam(), # 'adam',  # try different optimizers and different parameters!
                  # loss='binary_crossentropy',   # this is critical!
                  loss=fscore_loss,
                  metrics=['binary_accuracy', fscore])  
    model.summary()
    
    # decaying learning rate
    def step_decay(epoch):
        lrate = 0.1
        if epoch > 5:
            lrate = 0.1 / (2. ** epoch)
        return lrate

    lr = LearningRateScheduler(step_decay)
    callbacks_list = [lr]

    # now train it

    history = model.fit(data,     # training data
                        labels,   # training labels
                        class_weight = class_weights,
                        batch_size=256,
                        epochs=10,  # tried 1000 but it was too much!
                        verbose=1,
                        validation_data=
                        (validation_data,
                         validation_labels),
                        callbacks=callbacks_list
                        )

    basename = 'model-%d-%d-%d' % (
        single_genre_index,
        fc_size_1,
        fc_size_2)
    model.save(basename + '.h5')
    
    o = open(basename + 'history.txt', 'w')
    o.write(str(history.history))

    Predictions = (model.predict(data) >= .5).flatten() # class weight took care of imbalance
    TruePos = (Predictions & labels).sum()
    TrueNeg = ((1-Predictions) & (1-labels)).sum()
    FalsePos = (Predictions & (1-labels)).sum()
    FalseNeg = ((1-Predictions) & labels).sum()
    
    # Print confusion matrix
    o.write('Train conf matrix: %d %d %d %d\n' % (TruePos, TrueNeg, FalsePos, FalseNeg,))

    Precision = TruePos / (TruePos + FalsePos + 1e-10)
    Recall = TruePos / (TruePos + FalseNeg + 1e-10)
    F_Score = 2 * (Precision * Recall) / (Precision + Recall)

    o.write('Train p r f: %f %f %f\n' % (Precision, Recall, F_Score))

    # Compute F-score using validation data
    Predictions = (model.predict(validation_data) >= .5).flatten() # class weight took care of imbalance
    TruePos = (Predictions & validation_labels).sum()
    TrueNeg = ((1-Predictions) & (1-validation_labels)).sum()
    FalsePos = (Predictions & (1-validation_labels)).sum()
    FalseNeg = ((1-Predictions) & validation_labels).sum()
    
    # Print confusion matrix
    o.write('Train conf matrix: %d %d %d %d\n' % (TruePos, TrueNeg, FalsePos, FalseNeg,))

    Precision = TruePos / (TruePos + FalsePos + 1e-10)
    Recall = TruePos / (TruePos + FalseNeg + 1e-10)
    F_Score = 2 * (Precision * Recall) / (Precision + Recall)

    o.write('Train p r f: %f %f %f\n' % (Precision, Recall, F_Score))

for fc1_size in (256, 1024, 2048):
    for fc2_size in (256, 64, 1024):
        for single_genre_index in range(18):
            do_one_genre(single_genre_index, fc1_size, fc2_size)
