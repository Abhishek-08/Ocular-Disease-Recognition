from keras.preprocessing import image
import pandas as pd
import numpy as np

from tensorflow import keras
import os
import sys

from helper import read_params

def main():

    cla = sys.argv
    model_path = cla[1]
    batch_size = cla[2]
    print(model_path)
    # model_path = 'C:/UCD/Sem 2/Deep Learning/Project/Models/vgg16_model_hist_aug_1.h5'
    model = keras.models.load_model(model_path)
    params = read_params()
    test_path = params['test_path']
    img_width = params['img_height']
    img_height = params['img_height']
    batch_size = params['batch_size']
    # folder_path = test_path+'/Testing images/'
    images = []
    for img in os.listdir(folder_path):
        print(img)
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=batch_size)

    predictions_df = pd.DataFrame(zip(os.listdir(folder_path), classes), columns = ['Filename','Prediction'])

    predictions_df['Prediction'] = predictions_df['Prediction'].apply(lambda x : str(x))
    predictions_df['Prediction'] = np.where(predictions_df['Prediction']=='[0]', 'Disease','Normal')
    predictions_df['Prediction'].value_counts()

    predictions_df.to_csv('predictions.csv', index =False)


if __name__ == '__main__':
    main()
