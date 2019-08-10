import keras

from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from keras.models import Model, Sequential
from keras import models
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard


def get_model(shape):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)

    model = BatchNormalization()(inputlayer)
    model = Conv2D(16, (3, 3), activation='elu', kernel_initializer='glorot_normal')(model) # glorot_normal = xavier
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Flatten()(model)
    model = Dense(32, activation='elu')(model)
    model = Dropout(0.25)(model)

    # 11 because background noise has been taken out
    model = Dense(12, activation='softmax')(model)

    model = Model(inputs=inputlayer, outputs=model)

    return model




def get_mnist_model(shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))

    return model


def get_leightweight_cnn(shape):
    nclass = 12
    inp = Input(shape=shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Conv2D(8, kernel_size=2, activation='relu')(norm_inp)
    img_1 = Conv2D(8, kernel_size=2, activation='relu')(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Conv2D(16, kernel_size=3, activation='relu')(img_1)
    img_1 = Conv2D(16, kernel_size=3, activation='relu')(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Conv2D(32, kernel_size=3, activation='relu')(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Flatten()(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation='relu')(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation='relu')(dense_1))
    dense_1 = Dense(nclass, activation='softmax')(dense_1)

    return models.Model(inputs=inp, outputs=dense_1)


def get_vgg16_model(shape):
    input_tensor = Input(shape=(99, 161, 3))

    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block2_pool'].output

    # Stacking a new simple convolutional network on top of it
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(12, activation='softmax')(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    custom_model = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:7]:
        layer.trainable = False

    return custom_model