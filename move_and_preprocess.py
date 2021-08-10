from pathlib import Path
import shutil
import random
import os
import json

import pandas as pd
import numpy as np
import cv2

import sklearn
from sklearn.model_selection  import train_test_split

from helper import read_params


def create_label_directories(path, df):
    '''
    Function to create directories with label names in train, test and val_loss
    Input - Path where directory needs to created
    df  -  Dataframe containing mapping of files and its labels
    '''
    for i in set(df['labels']):
        Path(path+'/'+i).mkdir(parents=True, exist_ok=True)
    print("Folders for Labels created")

def move_files_to_label_directories(path,df,params):
    '''
    Function to move files to to the appropriate label directories as mentioned in the csv files
    Inputs :
    path - Base path of file
    df - Dataframe containing mapping of files and its labels
    params - params from params.json file
    '''
    for _,row in df.iterrows():
        for i in set(df['labels']):
            src_path = Path(params['train_path']+row['filename'])
            dest_path = Path(path+'/'+i+'/'+row['filename'])
            if dest_path.exists():
                pass
            else:
                if (row['labels']==i):
                    shutil.move(src_path,dest_path)
    print("Label directories created")
    print("Images were moved")

def move_test_images(path, test_images):
    '''
    Function to move test images to an inner directory
    path - test path
    test_images - list of test images
    '''
    Path(path+'/Testing images/').mkdir(parents=True, exist_ok=True)
    for i in test_images:
        shutil.move(path+'/'+i,path+'/Testing images/'+i)
    print("Test images were moved")

def move_images_to_superclass_folders(train_path,validation_path):
    '''
    Move individual diseases from its folders to the disease superclass
    Inputs :
    train_path - train path from params
    validation_path - validation path from params
    '''
    folders = ['A/.','C/.','D/.','G/.','H/.','M/.','O/.']
    for i in folders:
        shutil.copytree(train_path+i, train_path + 'Dis/.', dirs_exist_ok=True)
        shutil.copytree(validation_path+i, validation_path + 'Dis/.', dirs_exist_ok=True)

def hist_equal(file_name, input_path, output_path):
    '''
    Function to histogram equalize the histogram of colour images by splitting the 3 channels
    Inputs :
    file_name - Image name
    input_path - Input path
    output_path - Output path
    '''

    img = cv2.imread(input_path + file_name)
    R, G, B = cv2.split(img)

    output_red = cv2.equalizeHist(R)
    output_green = cv2.equalizeHist(G)
    output_blue = cv2.equalizeHist(B)

    equ = cv2.merge((output_red, output_green, output_blue))
    cv2.imwrite(output_path + file_name, equ)



def main():

    params = read_params()

    #read params file
    base_path = params['base_path']
    test_path = params['test_path']
    train_path = params['train_path']
    val_path = params['val_path']
    random_state = params['random_state']

    df = pd.read_csv('../full_df.csv')
    df['labels'] = df['labels'].apply(lambda x : x[2])

    print("Value Counts of Labels before roll up", df['labels'].value_counts())
    df1= df.copy()
    df1['labels'] = np.where(df1['labels']!='N','Disease','Normal')
    print("Value Counts of Labels after roll up", df1['labels'].value_counts())

    _y = df['labels']
    _X = df.loc[:, df.columns != 'labels']

    #Split training and validation from train with a random state of 42 for reproducibility and shuffle the data
    X_train, X_val, y_train, y_val = train_test_split(_X, _y, test_size=0.25, random_state=random_state, shuffle = True)

    train = X_train
    train['labels'] = y_train
    val = X_val
    val['labels'] = y_val
    val_images = list(val['filename'])

    #Creating Validation directory
    Path(val_path).mkdir(parents=True, exist_ok=True)

    create_label_directories(val_path, df)
    move_files_to_label_directories(val_path,val, params)

    train_images = [f for f in os.listdir(train_path) if f.endswith('.jpg')]
    test_images = [f for f in os.listdir(test_path) if f.endswith('.jpg')]

    create_label_directories(train_path, df)
    move_files_to_label_directories(train_path,train,params)
    move_test_images(test_path, test_images)
    move_images_to_superclass_folders(train_path,val_path)

    #Histogram equalisation
    Path(train_path+'Hist train/Dis').mkdir(parents=True, exist_ok=True)
    Path(val_path+'Hist val/Dis').mkdir(parents=True, exist_ok=True)
    Path(train_path+'Hist train/N').mkdir(parents=True, exist_ok=True)
    Path(val_path+'Hist val/N').mkdir(parents=True, exist_ok=True)

    print("Starting Histogram equalisation. Will take a while to complete")
    for i in ['Dis/','N/']:
      input_path = [train_path + i, val_path + i]
      output_path = [base_path + 'Training Images/Hist train/'+i,base_path + 'Validation Images/Hist val/'+i]
      for file_name in os.listdir(input_path[0]):
        print(file_name)
        hist_equal(file_name, input_path[0], output_path[0])
      for file_name in os.listdir(input_path[1]):
        print(file_name)
        hist_equal(file_name, input_path[1], output_path[1])
    print("Histogram equalisation Completed")

if __name__ == '__main__':
    main()
