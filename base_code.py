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

def read_images_as_tf_data(train_path, val_path, img_height, img_width, batch_size):
	'''
	Read images from train and val folders as tensorflow data

	Input :
	train_path - Folder path of train images
	val_path - Folder path of validation images
	img_height - Height of the image
	img_width - Width of the image
	batch_size - Size of the batch to be read from folders

	Output :
	train_ds - TF data containing train images of 2 classes
	val_ds - TF data containing validation images of 2 classes
	'''
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	  train_path,
	  seed=42,
	  image_size=(img_height, img_width),
	  batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	  val_path,
	  seed=42,
	  image_size=(img_height, img_width),
	  batch_size=batch_size)
	class_names = train_ds.class_names
	AUTOTUNE = tf.data.AUTOTUNE
	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
	return train_ds, val_ds, class_names

def data_aug(img_height, img_width):
	'''
	Function to define and return the Data Augmentation layer to be used in the network
	'''
	data_augmentation = tf.keras.Sequential(
	  [
	   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	   layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
	   layers.experimental.preprocessing.RandomRotation(0.1),
	   layers.experimental.preprocessing.RandomZoom(0.1)
	  ]
	)
	return data_augmentation

def base_model(data_augmentation):
	'''
	Function to build, compile and return the Base model using Keras Sequential layers

	Input :
	data_augmentation - The Data Augmentation layer

	Output :
	model - Convolutional neural network with a binary loss function
	'''
	model = Sequential([data_augmentation,
	  layers.Conv2D(32, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(32, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Conv2D(64, 3, padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Flatten(),
	  layers.Dense(128, activation='relu'),
	  layers.Dense(32, activation = 'relu'),
	  layers.Dense(1, activation = 'sigmoid')
	])

	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='binary_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy','AUC'])

	return model

def base_model_tuner(hp):
	'''
	Function to build, compile and return the Base model using Keras Sequential layers

	Input :
	hp - Hyperparameter object

	Output :
	hypermodel - Convolutional neural network with a binary loss function
	'''
	data_augmentation = data_aug()
	hypermodel = Sequential([data_augmentation,
					  layers.Conv2D(hp.Choice('num_filters_1', [32,64,128], default = 32),
									kernel_size = 3,
									padding = 'same',
									activation = 'relu'
									),
					  layers.MaxPooling2D(),
					  layers.Conv2D(hp.Choice('num_filters_2', [32,64,128,256], default = 32),
									kernel_size = 3,
									padding = 'same',
									activation = 'relu'
									),
					  layers.MaxPooling2D(),
					  layers.Conv2D(hp.Choice('num_filters_3', [32,64,128,256] , default = 64),
									kernel_size = 3,
									padding = 'same',
									activation = 'relu'
									),
					  layers.MaxPooling2D(),
					  layers.Flatten(),
					  layers.Dense(hp.Int('units_1', 32, 256, 32, default = 128),
								  activation = 'relu'),
					  layers.Dense(hp.Int('units_2', 32, 256, 32, default = 32),
								  activation = 'relu'),
					  layers.Dense(1, activation = 'sigmoid')
					  ])

	hypermodel.compile(
	  optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
	  ),
	  loss="binary_crossentropy",
	  metrics=["accuracy","AUC"],
	)
	return hypermodel


def main():
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

	train_ds, val_ds, class_names = read_images_as_tf_data(train_path, val_path, img_height, img_width, batch_size)

	plot_sample_images(train_ds, class_names)

	data_augmentation = data_aug(img_height, img_width)
	inp = input("Do you want hyper parameter tuning to decide the layers and filters of the base model? (Y/N) ? :")
	if str.lower(inp) == 'n':
		model = base_model(data_augmentation)
		print("Base model", model.summary())

		history = model.fit(
		  train_ds,
		  validation_data=val_ds,
		  epochs=35
		)

		plot_training_validation_scores(history, len(history.history['loss']), 'base_model_hist_aug_1.jpg')
		model.save(model_save_path + "base_model_hist_aug_1.h5")
	else:
		base_tuner = kt.Hyperband(
			base_model_tuner,
			objective=kt.Objective("val_auc", direction="max"),
			seed=42,
			max_epochs=5,
			directory=model_save_path + 'base_model/'
		)
		stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

		base_tuner.search(train_ds, epochs=20, validation_data=val_ds,  callbacks=[stop_early])

		best_hps=base_tuner.get_best_hyperparameters(num_trials=1)[0]
		model = base_tuner.hypermodel.build(best_hps)
		# learning_rate_reduction = ReduceLROnPlateau(monitor='val_auc',
		#                                             patience=5,
		#                                             verbose=1,
		#                                             factor=0.2,
		#                                             min_lr=0.00001)
		history = model.fit(train_ds, epochs = 30, validation_data=val_ds, verbose=1)

		plot_training_validation_scores(history, len(history.history['loss']), 'tuned_model_hist_aug_1.jpg')
		model.save(model_save_path + "tuned_model_hist_aug_1.h5")

if __name__ == '__main__':
	main()
