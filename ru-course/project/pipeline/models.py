import keras

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def get_model(shape):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)

    model = BatchNormalization()(inputlayer)
    model = Conv2D(16, (3, 3), activation='elu')(model)
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Flatten()(model)
    model = Dense(32, activation='elu')(model)
    model = Dropout(0.25)(model)

    # 11 because background noise has been taken out
    model = Dense(11, activation='sigmoid')(model)

    model = Model(inputs=inputlayer, outputs=model)

    return model