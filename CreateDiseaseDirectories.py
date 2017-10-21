import os
import shutil
from tqdm import *
import pandas as pd
import numpy as np

df =  pd.read_csv("Data_Entry_2017.csv") 

#print(df)

#os.makedirs("train")

#path to directory of 001 images
DATA_DIR = os.path.join("data")
TRAIN_DIR = os.path.join("data", "train")

#TEST_DIR = os.path.join()

FILENAME_COL = "Image Index"
CLASS_COL = "Finding Labels"


filenames = os.listdir(DATA_DIR)
filenames.sort()

for filename in tqdm(filenames):
    class_label = df[df[FILENAME_COL].str.match(filename)][CLASS_COL].values[0]
    if '|' in class_label:
        class_label = class_label.split('|')[0]

            
    class_path = os.path.join(TRAIN_DIR, class_label)
    
    #if directory for class doesnt exist, make it
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    
    shutil.move(os.path.join(DATA_DIR, filename), class_path)    

'''
#PATH_001 = os.path.join( DATA_DIR, "images", "images_001" ) 
#PATH_002 = os.path.join( DATA_DIR, "images", "images_002" ) 


#filenames = os.listdir(PATH_001)
#filenames2 = os.listdir(PATH_002)
#filenames.sort()
#filenames2.sort()
imagesPaths = [filenames, filenames2]

for images in imagesPaths:
    # loop through all filenames
    for filename in tqdm(images):

        #for every row in data frame
        for row in  range(len(df[FILENAME_COL])) :

            #find file in datafram
            if filename.endswith( df[FILENAME_COL][row] ):
                
                #get class of file
                class_label = df[CLASS_COL][row]
                if '|' in class_label:
                    class_label = class_label.split('|')[0]
                
                #make path for class label
                #class_path = os.path.join(DATA_DIR, class_label)
                
                class_path = os.path.join(TRAIN_DIR, class_label)
                
                #if directory for class doesnt exist, make it
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
        
                # move file from PATH_00x to class_path herei
                #print(os.path.join(PATH_001, filename))
                #print(class_path)
                shutil.copy(os.path.join(PATH_001, filename), class_path)    
'''

'''
path = os.path.join( DATA_DIR, "images", "images_001" ) 
#for each filename in the directory
for i in tqdm( range(len(   os.listdir(path)    ))  ):

        
    filename = os.path.join(path, os.listdir(path)[i])
    print(filename)

    #

    class_folders_path = os.path.join(".", "data")
    
    
    #loop through all image names in csv
    for row in tqdm( range(len( df["Image Index"]))  ):
        

        #find corresponding label in csv
        if filename.endswith( df["Image Index"][row]    ):
            

            #set bool for making dirs to False
            folderExists = False
            
            #check to see if label has a folder 
            for folder in os.listdir(class_folders_path):

               # if os.path.exists(df["Image Index"][row]):
                if folder == df["Finding Labels"][row]:
                    folderExists = True

                    break
                    #copy
                    #shutil.copy() 


            if not folderExists:
                os.makedirs(df["Finding Labels"][row])
                break            
'''


                

                
                
                
                    

                
                #if folder isnt there
                
'''
else if not os.path.exists(df["Image Index"][row]):
        
        #make the folder 
        os.makedirs(df["Image Index"][row])
   ''' 



'''
#if folder exists, set bool to True then leave
if folder == df["Image Index"][row]:
    folderExists = True    
    break

#if folder doesnt exist
if not folderExists:
            
'''
'''
            #if so add to the folder

            #if no folder, make one 

            #then add file to folder
            
            print( df["Image Index"][row]   )
            print( df["Finding Labels"][row]      )
            print()
            break;

'''

            #print("match")

       # print(  df["Image Index"][row]  )
        #print(filename)

#for i in tqdm( range(  5    )  ):

#for i in range(len(   os.listdir(path)    ))  :

    #check which row it is in csv

#if df["Image Index"][row] == filename:
        #if df[row].any() == filename:
  # ''' 
   #     print("Match")
    #    break;
   # '''
        #df["Finding Labels"] == 

        #if df["Image Index"][row] == filename:
         #   print("yes")
            #print(df["Finding Labels"][row])

        

#print(df["Image Index"][0])







