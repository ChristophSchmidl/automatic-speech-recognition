# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import Adam
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard

import os

from preprocessing import *
from models import *
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


#####################################################################
#      Visualize first sample from each training class
#####################################################################



# Give me the first occurence of every class as list: every item has word, path
sample_of_each_class = train_full.groupby('word', as_index=False).first().values

print(sample_of_each_class)





# Show me 9 samples using spectrograms
fig = plt.figure(figsize=(10, 10))

for idx, item in enumerate(sample_of_each_class[:9]):
    # Make subplots
    plt.subplot(3, 3, idx + 1)

    # pull the labels
    label = item[0]
    plt.title(label)

    # create spectrogram
    f, t, spectrogram = get_spectrogram(item[1])

    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.axis('off')

plt.show(block=False)




# Show me 9 samples using log spectrograms
fig = plt.figure(figsize=(10, 10))

for idx, item in enumerate(sample_of_each_class[:9]):
    # Make subplots

    plt.subplot(3, 3, idx + 1)

    # pull the labels
    label = item[0]
    plt.title(label)

    # create spectrogram
    f, t, spectrogram = get_log_spectrogram(item[1])

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')

plt.show(block=False)



#####################################################################
#      Visualize random sample from full training set
#####################################################################

train_full_arr = train_full.values
X_full = train_full_arr[:, 0]
y_full = train_full_arr[:, 1]


print("X_full:")
print(X_full)

print("y_full:")
print(y_full)


# Give me a random spectrogram and log_spectrogram

rand_input, rand_label = get_random_input_and_label(X_full, y_full)

print("Random input: {}".format(rand_input))
print("Random label: {}".format(rand_label))


_, _, random_spectrogram = get_spectrogram(rand_input)
_, _, random_log_spectrogram = get_log_spectrogram(rand_input)

fig = plt.figure()
plt.title(rand_label)
plt.imshow(random_spectrogram, aspect='auto', origin='lower')
plt.axis('off')
plt.show(block=False)

fig = plt.figure()
plt.title(rand_label)
plt.imshow(random_log_spectrogram.T, aspect='auto', origin='lower')
plt.axis('off')
plt.show(block=False)


print("Dimensions of spectrogram: {}".format(random_spectrogram.shape))
print("Dimensions of log spectrogram: {}".format(random_log_spectrogram.shape))



#####################################################################
#      Visualize random sample from reduced training set (kaggle valid)
#####################################################################

train_reduced_arr = train_reduced.values
X_reduced = train_reduced_arr[:, 0]
y_reduced = train_reduced_arr[:, 1]


print("X_reduced:")
print(X_reduced)

print("y_full:")
print(y_reduced)


# Give me a random spectrogram and log_spectrogram

rand_input, rand_label = get_random_input_and_label(X_reduced, y_reduced)

print("Random input: {}".format(rand_input))
print("Random label: {}".format(rand_label))


_, _, random_spectrogram = get_spectrogram(rand_input)
_, _, random_log_spectrogram = get_log_spectrogram(rand_input)

fig = plt.figure()
plt.title(rand_label)
plt.imshow(random_spectrogram, aspect='auto', origin='lower')
plt.axis('off')
plt.show(block=False)

fig = plt.figure()
plt.title(rand_label)
plt.imshow(random_log_spectrogram.T, aspect='auto', origin='lower')
plt.axis('off')
plt.show(block=False)


print("Dimensions of spectrogram: {}".format(random_spectrogram.shape))
print("Dimensions of log spectrogram: {}".format(random_log_spectrogram.shape))




#####################################################################
#                           Model Training
#####################################################################

spectrogram_shape = (129, 124, 1)
log_spectrogram_shape = (99, 161, 1)


model = get_leightweight_cnn(log_spectrogram_shape)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print(model.summary())

labelbinarizer = LabelBinarizer()
X = X_reduced
y = labelbinarizer.fit_transform(y_reduced)
X, Xt, y, yt = train_test_split(X, y, test_size=0.3, stratify=y)


temp_batch = batch_generator(X, y, batch_size=32)

#print(temp_batch)

#print("Dimensions of a batch: {}".format(temp_batch[0].shape))


tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)



train_gen = batch_generator(X, y, batch_size=32)
valid_gen = batch_generator(Xt, yt, batch_size=32)

model.fit_generator(
    generator=train_gen,
    epochs=30,
    steps_per_epoch=X.shape[0] // 32,
    validation_data=valid_gen,
    validation_steps=Xt.shape[0] // 32,
    use_multiprocessing=False,
    workers=1,
    callbacks=[tensorboard])




#####################################################################
#                       Make predicitions
#####################################################################


print("Calculating predictions...")

start = time.time()

predictions = []
paths = test.path.tolist()


for path in paths:
    specgrams = []
    _, _, specgram = get_log_spectrogram(path)
    specgram = specgram.reshape(99, 161, -1)
    specgrams.append(specgram)
    pred = model.predict(np.array(specgrams))
    #print(pred)
    #argmax = np.argmax(pred, axis=1)
    #print("Argmax: {}".format(argmax))

    #label = labelbinarizer.classes_[argmax]
    #print(label)
    predictions.append(pred)

end = time.time()
print("Calculating predicitions time: {}".format(end - start))


print("Transforming predicitions to labels and writing csv...")


labels = [labelbinarizer.classes_[np.argmax(p, axis=1)][0] for p in predictions]
test['labels'] = labels
test.path = test.path.apply(lambda x: str(x).split('/')[-1])
submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
submission.to_csv('simple-keras-model-with-data-generator_submission_new.csv', index=False)

