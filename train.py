import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from utils.normalize import naive_normalization
from models.neuralNetwork import convNeuralNet as cnn
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from tqdm import *

TRAIN_DIR = os.path.join('data', 'train')
EXPECTED_SHAPE = (1024,1024)

def now():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def load_data():
    dirList = [os.path.join(TRAIN_DIR, o) for o in os.listdir(TRAIN_DIR)\
            if os.path.isdir(os.path.join(TRAIN_DIR, o))]

    imageList = []
    imageLabels = []
    for directory, disease_idx in zip(dirList, range(len(dirList))):
        LIMIT = 75 
        filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        filenames.sort()
        filenames = filenames[:LIMIT]
        for filename in tqdm(filenames):
            tmp_img = np.array(Image.open(filename))
            if tmp_img.shape != EXPECTED_SHAPE:
                continue
            tmp_img = naive_normalization(tmp_img)
            imageList.append(tmp_img)
            imageLabels.append(disease_idx)


    return np.array(imageList), np.array(imageLabels)


######### LOAD DATA ###########

X, y = load_data()
y = to_categorical(y)

from sklearn.utils import shuffle
X,y = shuffle(X,y)

print ('X shape', X.shape)
print ('y shape', y.shape)


######### LOAD MODEL ###########
LR = 1e-3
model = cnn(input_shape=X[0].shape, num_classes = y.shape[1], lrate=LR)


######### CHECKPOINTS ###########
WEIGHT_DIR = "weights"
if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
weight_path = now() + "-{epoch:02d}-{val_acc:.2f}.hdf5"
weight_path = os.path.join(WEIGHT_DIR, weight_path)
checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True,\
                            save_weights_only=True, mode='max')
callbacks_list = [checkpoint]


######### TRAIN MODEL ###########

BATCH_SIZE = 2
N_EPOCHS = 10000
model.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=callbacks_list, validation_split=0.2) 
