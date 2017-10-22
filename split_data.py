import os
import random
import shutil
from tqdm import *

SRC_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")
TEST_DIR = os.path.join("data", "test")

ALL_DIRS = os.listdir(SRC_DIR)

for directory in ALL_DIRS:
    filenames = os.listdir(os.path.join(SRC_DIR,directory))
    filenames.sort()

    for filename in tqdm(filenames):
        roll = random.uniform(0,1)
        if roll < 0.2: 
            if not os.path.exists(os.path.join(VAL_DIR, directory)):
                os.makedirs(os.path.join(VAL_DIR, directory))
            shutil.move(os.path.join(SRC_DIR, directory, filename),\
                    os.path.join(VAL_DIR, directory, filename))  
            continue
        roll = random.uniform(0,1)
        if roll < 0.2: 
            if not os.path.exists(os.path.join(TEST_DIR, directory)):
                os.makedirs(os.path.join(TEST_DIR, directory))
            shutil.move(os.path.join(SRC_DIR, directory, filename),\
                    os.path.join(TEST_DIR, directory, filename))  
            
