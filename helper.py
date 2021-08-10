import json
import matplotlib.pyplot as plt
from pathlib import Path


def read_params():
    '''
    Helper function to read param.json file

    Output :
    params : params dictionary
    '''
    with open("params.json") as f:
        params = json.loads(f.read())
    return params

def plot_sample_images(train_ds, class_names):

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    Path('../Figures/').mkdir(parents=True, exist_ok=True)
    plt.savefig('../Figures/Images after histogram equalisation.jpg')

def plot_training_validation_scores(history, epochs, filename):

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  auc = history.history['auc']
  val_auc = history.history['val_auc']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Loss')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, auc, label='Training AUC')
  plt.plot(epochs_range, val_auc, label='Validation AUC')
  plt.legend(loc='upper right')
  plt.title('Training and Validation AUC')
  plt.show()
  Path('../Figures/').mkdir(parents=True, exist_ok=True)
  plt.savefig('../Figures/' + filename)
