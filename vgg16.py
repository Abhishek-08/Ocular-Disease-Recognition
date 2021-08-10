import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import applications, optimizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import kerastuner as kt

from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import pandas as pd

import matplotlib.pyplot as plt

from helper import read_params, plot_sample_images, plot_training_validation_scores
from base_code import read_images_as_tf_data, data_aug

def base_vgg_model(data_augmentation):
    '''
    Function to download the VGG16 model (with imagenet weights) and build it by freezing all layers except the top layers

    Input :
    data_augmentation - The Data Augmentation layer

    Output :
    model - VGG16 with a binary loss function
    '''
    vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))
    for layer in vgg.layers:
        layer.trainable = False

    model = Sequential([data_augmentation,
                        vgg,
                        layers.Flatten(),
                        layers.Dense(128, activation='relu'),
                        layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy','AUC'])
    return model


def vgg_model_builder(hp):
    '''
    Second round of fine-tuning the inception model by only tuning the learning rate.

    Inputs :
    hp - Hyperparameter object

    Outputs :
    model - VGG16 model fine-tuned and compiled with binary_crossentropy loss
    '''
    model = keras.models.load_model(model_save_path + "vgg16_model_hist_aug_1.h5")
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=hp_learning_rate),
              metrics=['accuracy','AUC'])
    return model


def main():

    #Read and store params values
    params = read_params()

    global base_path, test_path, train_path, val_path, model_save_path, random_state, batch_size, img_height, img_width
    base_path = params['base_path']
    test_path = params['test_path']
    train_path = params['train_path_hist']
    val_path = params['val_path_hist']
    model_save_path = params["model_save_path"]

    random_state = params['random_state']
    batch_size = params['batch_size']
    img_height = params['img_height']
    img_width = params['img_width']

    print("Params read complete")
    train_ds, val_ds , class_names = read_images_as_tf_data(train_path, val_path, img_height, img_width, batch_size)
    print("Train and validation read complete")

    print("Starting VGG 16 model training")

    #Retrieve the data augmentation layer
    data_augmentation = data_aug(img_height, img_width)

    model = base_vgg_model(data_augmentation)

    #Save the max val_auc at multiple checkpoints
    checkpoint = ModelCheckpoint(model_save_path + "vgg16_model_hist_aug_1.h5",monitor='val_auc', mode ='max',
                                verbose=1,save_best_only=True, save_weights_only=False,period=1)

    #Early stopping when val_accuracy doesn't improve over a period of 5 epochs
    earlystop = EarlyStopping(monitor="val_accuracy",patience=5,verbose=1)
    history = model.fit(train_ds, epochs = 10, validation_data=val_ds, verbose=1, callbacks = [checkpoint,earlystop])

    print("First round of VGG 16 training complete")
    plot_training_validation_scores(history, len(history.history['loss']), "vgg16_model_hist_aug_1.jpg")

    print("Begin VGG16 hyperparameter tuning")
    # Using Keras
    tuner = kt.Hyperband(vgg_model_builder,
                         objective=kt.Objective("val_auc", direction="max"),
                         max_epochs=5,
                         directory=model_save_path+'VGG1/')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5)

    tuner.search(train_ds, epochs=20, validation_data=val_ds, callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_hps)
    checkpoint = ModelCheckpoint(model_save_path + "tuned_vgg16_model_hist_aug_1.h5",monitor="val_auc",mode = 'max',verbose=1,save_best_only=True,
                                 save_weights_only=False,period=1)
    earlystop = EarlyStopping(monitor="val_accuracy",mode="max",patience=10,verbose=1)
    history = model.fit(train_ds, epochs = 20, validation_data=val_ds, verbose=1, callbacks = [checkpoint,earlystop])

    print("First round of hyper parameter optimization complete")
    plot_training_validation_scores(history, len(history.history['loss']))

    # Second round of tuning but this time fixing the learning rate to 1e-5 because the model needs to learn slower to converge
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-5),
                  metrics=['accuracy','AUC']
                  )
    checkpoint = ModelCheckpoint(model_save_path + "tuned_vgg16_model_hist_aug_2.h5",monitor="val_auc",mode = 'max',verbose=1,save_best_only=True,
                                 save_weights_only=False,period=1)
    history = model.fit(train_ds, epochs = 20, validation_data=val_ds, verbose=1, callbacks = [checkpoint,earlystop])

    plot_training_validation_scores(history, len(history.history['loss']),"tuned_vgg16_model_hist_aug_2.jpg")

if __name__ == '__main__':
    main()
