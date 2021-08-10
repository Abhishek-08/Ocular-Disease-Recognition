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
from base_code import read_images_as_tf_data

def inception_tuner(hp):
	'''
	Function to build the best hyperparameters for the last layer of an inception model.
	This version will be used as the base for the inception model.

	Inputs :
	hp - Hyperparameter object

	Outputs :
	model - Inception model fine-tuned and compiled with binary_crossentropy loss
	'''
	#Downloading inception-v3 with imagenet weights
	base_model = applications.InceptionV3(weights='imagenet',
								  include_top=False,
								  input_shape=(img_height, img_width,3))
	trial = base_model.output

	#GlobalAveragePooling is to minimize overfitting
	trial = GlobalAveragePooling2D()(trial)

	# Adding a fully-connected layer. Adding various hyper parameter choices for number of filters
	trial = Dense(hp.Int('units_1', 32, 256, 32, default = 128), activation= 'relu')(trial)
	trial = Dense(hp.Int('units_2', 32, 256, 32, default = 64), activation= 'relu')(trial)

	#Since it's a binary class classification, we perform sigmoid activation
	predictions = Dense(1)(trial)
	predictions = Activation('sigmoid')(predictions)

	model = Model(inputs=base_model.input, outputs=predictions)

	#Freezing the bottom layers of the inception model
	for layer in base_model.layers:
	  layer.trainable = False

	#Compiling the model with various choices of learning rate
	model.compile(loss='binary_crossentropy',
				optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),
				metrics=['accuracy','AUC'])
	return model



def inception_model_builder(hp):
	'''
	Second round of fine-tuning the inception model by just tuning the learning rate.

	Inputs :
	hp - Hyperparameter object

	Outputs :
	model - Inception model fine-tuned and compiled with binary_crossentropy loss
	'''
	model = keras.models.load_model(model_save_path + "inception_model_hist_aug_1.h5")

	#Providing choices for tuning the learning rate
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

	print("Starting Transfer learning - Inception v3")
	# Tune inception v3 by choosing the best number of filters via HyperBand hyperparameter tuning
	inceptiontuner = kt.Hyperband(
		inception_tuner,
		objective = kt.Objective("val_auc", direction="max"),
		seed = 42,
		max_epochs = 10,
		directory = model_save_path+'Inception1/'
	)
	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

	inceptiontuner.search(train_ds, epochs=20, validation_data=val_ds,  callbacks=[stop_early])

	#Retrieve best hyper parameters
	best_hps = inceptiontuner.get_best_hyperparameters(num_trials=1)[0]
	model = inceptiontuner.hypermodel.build(best_hps)

	#Checkpoint the model with the highest validation AUC and perform early stopping when the model is not learning enough
	checkpoint = ModelCheckpoint(model_save_path + "inception_model_hist_aug_1.h5",monitor="val_auc",mode = 'max',verbose=1,save_best_only=True,
								 save_weights_only=False,period=1)
	earlystop = EarlyStopping(monitor="val_accuracy",patience=5,verbose=1)

	history = model.fit(
			train_ds,
			epochs=20,
			validation_data=val_ds,
			verbose=1,
			callbacks=[checkpoint,earlystop])

	plot_training_validation_scores(history,  len(history.history['loss']), 'inception_model_hist_aug_1.jpg')

	# Seond round of hyperparameter optimization to tune the learning rate of the first model
	tuner = kt.Hyperband(inception_model_builder,
						 objective=kt.Objective("val_auc", direction="max"),
						 max_epochs=5,
						 directory=model_save_path+'Inception1/')

	stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5)

	tuner.search(train_ds, epochs=20, validation_data=val_ds, callbacks=[stop_early])

	best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

	# Similar steps as the first round of hyper parameter optimization
	model = tuner.hypermodel.build(best_hps)
	checkpoint = ModelCheckpoint(mmodel_save_path + "tuned_inception_model_hist_aug_1.h5",monitor="val_auc",mode = 'max',verbose=1,save_best_only=True,
								 save_weights_only=False,period=1)
	earlystop = EarlyStopping(monitor="val_accuracy",patience=5,verbose=1)
	history = model.fit(train_ds, epochs = 15, validation_data=val_ds, verbose=1, callbacks = [checkpoint,earlystop])

	plot_training_validation_scores(history, len(history.history['loss']), 'tuned_inception_model_hist_aug_1.jpg')

if __name__ == '__main__':
	main()
