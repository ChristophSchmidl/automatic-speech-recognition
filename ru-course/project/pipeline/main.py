# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase

import time

import numpy as np
import pandas as pd

import os

from preprocessing import *
from models import *
from visualization import *
from DataGenerator import DataGenerator





#####################################################################
#                       Specify Globals
#####################################################################

train_audio_path = '../data/train/audio/'
test_audio_path = '../data/test/audio/'
samples = []
#train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


#####################################################################
#               Load ready to use Data as Dataframes
#####################################################################

# Pandas dataframe with columns: 'path', 'word'
# Prepare_data marks every other word out of the 10 train_words as "unknown"

train_full = get_data(train_audio_path)
train_reduced = prepare_data(get_data(train_audio_path))
test = prepare_data(get_data(test_audio_path))

print("Size, Shape and dimension of train_full df: {}, {}, {}".format(train_full.size, train_full.shape, train_full.ndim))
print("Size, Shape and dimension of train_reduced df: {}, {}, {}".format(train_reduced.size, train_reduced.shape, train_reduced.ndim))
print("Size, Shape and dimension of test df: {}, {}, {}".format(test.size, test.shape, test.ndim))

# Size, Shape and dimension of train df: 129454, (64727, 2), 2
# Size, Shape and dimension of test df: 317076, (158538, 2), 2


#####################################################################
#      Printing class distribution of train_full and train_reduced
#####################################################################


# Normalize=False -> Give me absolute counts
# Normalize=True -> Give me percentages
#print(train_full.word.value_counts(normalize=False))
#print(train_reduced.word.value_counts(normalize=False))


# visualize(train_full, train_reduced)




#####################################################################
#                           Model Training
#####################################################################

train_reduced_arr = train_reduced.values
X_reduced = train_reduced_arr[:, 0]
y_reduced = train_reduced_arr[:, 1]

spectrogram_shape = (129, 124, 1)
log_spectrogram_shape = (99, 161, 1)
log_spectrogram_shape_3d = (99, 161, 3)




train_and_predict("inception_v3", log_spectrogram_shape_3d, 10, X_reduced, y_reduced, test, 32, "imagenet")
