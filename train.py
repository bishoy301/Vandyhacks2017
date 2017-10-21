import os
import numpy as np
import glob as gb
from PIL import Image
from keras.utils import to_categorical

imageFolderPath = os.path.join('data', 'train')

def imageLabeling():
    dirList = [os.path.join(imageFolderPath, o) for o in os.listdir(imageFolderPath) if os.path.isdir(os.path.join(imageFolderPath, o))]

    for i, disease_idx in zip(dirList, range(len(dirList))):
        imgPath = os.path.join(imageFolderPath, i)
        imageList = np.array([np.array(Image.open(img)) for img in imgPath])
        imageLabels = np.array([i for x in range(len(dirList))])

    return imageList, imageLabels

