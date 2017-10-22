import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from utils.normalize import naive_normalization
from models.neuralNetwork import convNeuralNet as cnn
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from tqdm import *

TRAIN_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')
EXPECTED_SHAPE = (1024,1024)

def now():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def load_data():
    dirList = [os.path.join(TRAIN_DIR, o) for o in os.listdir(TRAIN_DIR)\
            if os.path.isdir(os.path.join(TRAIN_DIR, o))]

    imageList = []
    imageLabels = []
    for directory, disease_idx in zip(dirList, range(len(dirList))):
        filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]
        filenames.sort()
        if len(filenames) < 300:
            continue
        filenames = filenames[:750]
        for filename in tqdm(filenames):
            tmp_img = np.array(Image.open(filename),dtype=np.float16)
            if tmp_img.shape != EXPECTED_SHAPE:
                continue
            tmp_img = naive_normalization(tmp_img)
            imageList.append(tmp_img)
            imageLabels.append(disease_idx)


    return np.array(imageList,dtype=np.float16), np.array(imageLabels, dtype=np.float16)


######### LOAD DATA ###########

'''
X, y = load_data()
y = to_categorical(y)

from sklearn.utils import shuffle
X,y = shuffle(X,y)

print ('X shape', X.shape)
print ('y shape', y.shape)
'''



######### CHECKPOINTS ###########
WEIGHT_DIR = "weights"
if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
weight_path = now() + "-{epoch:02d}-{val_acc:.2f}.hdf5"
weight_path = os.path.join(WEIGHT_DIR, weight_path)
checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True,\
                            save_weights_only=False, mode='max')
callbacks_list = [checkpoint]


reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, min_lr=1e-6) 
callbacks_list.append(reduce_lr)

######### LOAD MODEL ###########
LOAD_WEIGHTS = True
LR = 1e-2
best_weights = os.listdir(WEIGHT_DIR)
best_weights.sort()
if LOAD_WEIGHTS:
    best_weights = os.path.join(WEIGHT_DIR, best_weights[-1])
else:
    best_weights = None

model = cnn(input_shape=(1024,1024,1), num_classes = 15, lrate=LR, weights=best_weights)

######### TRAIN MODEL ###########

BATCH_SIZE = 32
N_EPOCHS = 10000

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(1024,1024),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(1024,1024),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode="categorical")

model.fit_generator(
        train_generator,
        steps_per_epoch = 2000//BATCH_SIZE,
        epochs = N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=800//BATCH_SIZE,
        callbacks=callbacks_list)

