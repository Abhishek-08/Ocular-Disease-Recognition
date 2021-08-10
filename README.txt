README

Folder structure :

Project
|- README.txt
|- report.pdf
|- requirements.txt - Contains the requirments to be installed (pip install -r requirements.txt)
|- Codes
	|- base_code.py 
	|- eval.py 
	|- helper.py 
	|- inceptionv3.py 
	|- move_and_preprocess.py 
	|- vgg16.py 
	|- params.json - Contains the parameters
	
|- Models 
	|- base_model_hist_aug_1.h5 (Base model)
	|- tuned_model_hist_aug_1.h5 (Base model + hyperparameter tuning)
	|- inception_model_hist_aug_1.h5 (inception v3)
	|- tuned_inception_model_hist_aug_1.h5 (inception v3 + hyperparameter tuning)
	|- vgg16_model_hist_aug_1.h5 (vgg16 model)
	|- tuned_vgg16_model_hist_aug_1.h5 (vgg16 + hyperparameter tuning 1)
	|- tuned_vgg16_model_hist_aug_2.h5 (vgg16 + hyperparameter tuning 2 - Best performing model)
	
|- Figures
	|- Figures saved from codes
	
|- ODIR-5K
	|- ODIR-5K
		|- Testing Images
			|- Testing Images
		|- Training Images
			|- A
			|- C
			|- D
			|- Dis
			|- G
			|- H 
			|- Hist train 
				|- Dis
				|- N
			|- M 
			|- N 
			|- O
		|- Validation Images
			|- A
			|- C
			|- D
			|- Dis
			|- G
			|- H 
			|- Hist val 
				|- Dis
				|- N
			|- M 
			|- N 
			|- O
			
Python Codes :

1. move_and_preprocess.py - Contains functions to replicate the folder structures and moves files to train, test and validation 
after splitting and shuffling of data. Preprocessing images like resizing and histogram equalization is performed here

How to run? python move_and_preprocess.py

2. helper.py - Contains functions which support the other python files. Reading params and plotting images of the results and data is 
present in the helper files

3. base_code.py - Contains functions to preprocess, augment images and to create the base neural network and tune it

4. inceptionv3.py - Contains functions to freeze layers of inception v3 and train the top layer and tune the hyperparameters

5. vgg16.py - Contains functions to freeze layers of vgg16 and train the top layer and tune the hyperparameters

6. eval.py - Contains a function to predict a folder of images using an already saved model

How to run? - python eval.py "path_to_saved_model.h5" "path_to_directory_full_of_images"
eg : python eval.py "../Models/vgg16_model_hist_aug_1.h5" "../ODIR-5K/ODIR-5K/Testing Images/Testing Images/"

