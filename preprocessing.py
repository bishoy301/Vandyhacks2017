import numpy as np
import os
import random
import shutil
from tqdm import *
from PIL import Image

TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")
TEST_DIR = os.path.join("data", "test")
GRBG_DIR = os.path.join("data", "grbge")

if not os.path.exists(GRBG_DIR):
    os.makedirs(GRBG_DIR)

classes = os.listdir(TRAIN_DIR)

ALL_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]
for main_dir in ALL_DIRS:
    for directory in classes:
        filenames = os.listdir(os.path.join(main_dir,directory))
        filenames.sort()

        for filename in tqdm(filenames):
            img = np.array(Image.open(os.path.join(main_dir, directory, filename)))
            if len(img.shape) != 2:
                shutil.move(os.path.join(main_dir, directory, filename),\
                        os.path.join(GRBG_DIR, filename))
