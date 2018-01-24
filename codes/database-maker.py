#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:50:37 2018

@author: bob
"""

import logging as log
import os
import csv
import random
import numpy as np

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy import ndimage
import scipy.misc as sm

#%%
#Parameters
log.basicConfig(filename='logfile.log',level=log.DEBUG)

_data_dir = "/home/bob/Desktop/code/GalaxyZoo/Data/"
_csv_path = _data_dir+"training_solutions_rev1.csv"
_images_dir = _data_dir+"images_training_rev1/"
_db_path = _data_dir+"V4-96p/"

_class_num = 20 # Index of the class in CSV file to check
_class_treshold = [0.3,1] # Range of the acceptable value in _class_num item

_selected_ratio = 0.33 # Percentage of TRUE values in final DataSet

_test_ratio = 0.33 # Percentage of the data must be used as test

_image_size = [96,96]
#%%
#Functions

# Get a bunch of ids and a label, fetch the image from file and make it grayscal
# then append it in _imgs array and reshape it. then return the whole things
def packImagesWithLabels(_ids_db):
    
    _imgs = []
    _labels = []
    for item in _ids_db:
        _id = item[0]
        _label = item[1]
        log.debug("Entered packImagesWithLabels with label of {}".format(_label))
        _im = ndimage.imread(_images_dir+str(_id)[:]+".jpg")
        _im = sm.imresize(_im,size=_image_size)
        
        _im = np.sum(_im,axis=2).reshape(_im.shape[0],_im.shape[1],1)
        _imgs.append(_im)
        _labels.append(_label)
    _imgs = np.array(_imgs)
    _imgs = np.reshape(_imgs, (_ids_db.shape[0],_im.shape[0],_im.shape[1],1))
    return _imgs,_labels

# This function will check if the _class with the index of _class_num is in range
# of the interest. The range has been setted before in _class_treshold
# if the data has been recived from CSVRead function, use index 1 insted of _class_num
def checkCSVRow(_row,_class_num=_class_num,_range=_class_treshold):
    #log.debug("checking csv row class to be in range")
    if float(_row[_class_num]) > _range[0] and float(_row[_class_num]) <= _range[1]:
        log.info("id: {} in range of {}:{}".format(_row[0],_range[0],_range[1]))
        return True
    else:
        return False
    

# This function will read the CSV file and return the contents of id (index 0) and _class_num
def CSVRead(_path):
    with open(_path) as csvfile:
        _cdata = csv.reader(csvfile,delimiter=',')
        _ = []
        for _row in _cdata:
            _.append([_row[0],_row[_class_num]])
    csvfile.close()
    return np.array(_[1:])

# select numbers of false values from _cdata according to _selected_ratio
def falseValueSelection(_cdata,_indxs,_selected_count,_selected_ratio=_selected_ratio):
    _false_count = int((_selected_count/_selected_ratio) - _selected_count)
    _cdata = np.delete(_cdata,_indxs)
    _false_value_ids = np.random.choice(_cdata,_false_count)
    return _false_value_ids

# get two list of images and labels, merge and then shuffels them
def mergeSelections(_true_ids,_false_ids):
    _ids_db = []
    for item in _true_ids:
        _ids_db.append([item,1])
    for item in _false_ids:
        _ids_db.append([item,0])
    np.random.shuffle(_ids_db)
    return np.array(_ids_db)
    
#%%
log.info("starting")
_cdata = CSVRead(_csv_path)
_selected_ids = []
_indxs = []
_i = 0

# check in CSV data to find the records in which their _class_num value is in range of _class_treshold
# and save thire ids in _ids
for _row in _cdata:
    if checkCSVRow(_row,1):
        _selected_ids.append(_row[0])
        _indxs.append(_i)
    _i+=1

# remove selected ids from _cdata and select False records according to _selected_ratio
_false_ids = falseValueSelection(_cdata[:,0],_indxs,np.size(_selected_ids))

# merge two packes and shuffle
_ids_db = mergeSelections(_true_ids=_selected_ids,_false_ids=_false_ids)

#make final database in RAM
_imgs_DB , _labels_DB = packImagesWithLabels(_ids_db=_ids_db)
#%%
x = _imgs_DB
y = _labels_DB
Y = np_utils.to_categorical(y, 2) # Make Categorical labels: 1 = [0,1] & 0 = [1,0]

# split to train/test

print("making train test split")
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=_test_ratio)
print("Done")

# Write on the disk

# Check if folders exist
if not os.path.exists(_db_path+"train/"):
    os.makedirs(_db_path+"train/")
if not os.path.exists(_db_path+"test/"):
    os.makedirs(_db_path+"test/")
        
np.save(_db_path+"train/x",X_train)
np.save(_db_path+"train/Y",y_train)
np.save(_db_path+"test/x",X_test)
np.save(_db_path+"test/Y",y_test)


#%%
