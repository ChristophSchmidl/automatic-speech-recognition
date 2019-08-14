import keras

import numpy as np

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras import models
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import time

from sklearn.preprocessing import LabelBinarizer

from preprocessing import *


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
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
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


def get_vgg16_model(shape, weights='imagenet'):
    # Load the VGG model
    vgg_conv = VGG16(weights=weights, include_top=False, input_shape=shape)

    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers:
        layer.trainable = True

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)


    # Add new layers
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model


def get_resnet50_v2_model(shape, weights='imagenet'):
    # Load the RestNet50 V2 model
    restnet50v2_conv = ResNet50(weights=weights, include_top=False, input_shape=shape)

    # Freeze the layers except the last 4 layers
    for layer in restnet50v2_conv.layers:
        layer.trainable = True

    # Check the trainable status of the individual layers
    for layer in restnet50v2_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(restnet50v2_conv)

    # Add new layers
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model

def get_inception_v3_model(shape, weights='imagenet'):
    # Load the Inception model
    inceptionv3_conv = InceptionV3(weights=weights, include_top=False, input_shape=shape)

    # Freeze the layers except the last 4 layers
    for layer in inceptionv3_conv.layers:
        layer.trainable = True

    # Check the trainable status of the individual layers
    for layer in inceptionv3_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(inceptionv3_conv)

    # Add new layers
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model


def get_resnet_model(shape):
    pass

def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1):
    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding='same', name='block{}_conv{}'.format(block, layer))(inp)
        x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        x = Dropout(dropout, name='block{}_dropout{}'.format(block, layer))(x)
        return x

    return layer_wrapper


def dense_block(units, dropout=0.2, activation='relu', name='fc1'):
    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper


def get_vgg16_bn(input_tensor=None, input_shape=None, classes=1000, conv_dropout=0.1, dropout=0.3, activation='relu'):
    """Instantiates the VGG16 architecture with Batch Normalization
    # Arguments
        input_tensor: Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: shape tuple
        classes: optional number of classes to classify images
    # Returns
        A Keras model instance.
    """
    img_input = Input(shape=input_shape) if input_tensor is None else (
        Input(tensor=input_tensor, shape=input_shape) if not K.is_keras_tensor(input_tensor) else input_tensor
    )

    # Block 1
    x = conv_block(32, dropout=conv_dropout, activation=activation, block=1, layer=1)(img_input)
    x = conv_block(32, dropout=conv_dropout, activation=activation, block=1, layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=2, layer=1)(x)
    x = conv_block(64, dropout=conv_dropout, activation=activation, block=2, layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=3, layer=1)(x)
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=3, layer=2)(x)
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=3, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=4, layer=1)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=4, layer=2)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=4, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=5, layer=1)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=5, layer=2)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=5, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Flatten
    x = GlobalAveragePooling2D()(x)

    # FC Layers
    x = dense_block(512, dropout=dropout, activation=activation, name='fc1')(x)
    x = dense_block(512, dropout=dropout, activation=activation, name='fc2')(x)

    # Classification block
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    inputs = get_source_inputs(input_tensor) if input_tensor is not None else img_input

    # Create model.
    return Model(inputs, x, name='vgg16_bn')



def train_and_predict(model_name, shape, num_epochs, X, y, test, batch_size, weights='imagenet'):

    model = None

    if model_name is "vgg16":
        model = get_vgg16_model(shape, weights=weights)
    if model_name is "inception_v3":
        model = get_inception_v3_model(shape, weights=weights)
    if model_name is "resnet50_v2":
        model = get_resnet50_v2_model(shape, weights=weights)
    if model_name is "leightweight_cnn":
        model = get_leightweight_cnn(shape)
    if model_name is "mnist":
        model = get_mnist_model(shape)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    print(model.summary())

    labelbinarizer = LabelBinarizer()
    X = X
    y = labelbinarizer.fit_transform(y)

    X, Xt, y, yt = train_test_split(X, y, test_size=0.3, stratify=y)

    # temp_batch = batch_generator(X, y, batch_size=32)

    # print(temp_batch)

    # print("Dimensions of a batch: {}".format(temp_batch[0].shape))

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=batch_size)

    train_gen = None
    valid_gen = None

    if model_name in ["vgg16", "inception_v3", "resnet50_v2"]:
        train_gen = batch_generator(X, y, batch_size=batch_size, num_channels=3)
        valid_gen = batch_generator(Xt, yt, batch_size=batch_size, num_channels=3)
    else:
        train_gen = batch_generator(X, y, batch_size=batch_size)
        valid_gen = batch_generator(Xt, yt, batch_size=batch_size)

    #callbacks = [EarlyStopping(monitor='val_loss', patience=3),
    #             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    #             tensorboard]

    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                 tensorboard]

    training_start = time.time()


    history = model.fit_generator(
        generator=train_gen,
        epochs=num_epochs,
        steps_per_epoch=X.shape[0] // batch_size,
        validation_data=valid_gen,
        validation_steps=Xt.shape[0] // batch_size,
        use_multiprocessing=False,
        workers=1,
        callbacks=callbacks,
        verbose=1)

    training_end = time.time()
    print("Training time: {}".format(training_end - training_start))

    print("Loading best model...")
    model.load_weights('best_model.h5')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print("History:")

    print(history.history)


    best_epoch_idx = np.argmax(history.history['val_acc'])

    print("Epoch with the highest val acc: {}".format(best_epoch_idx))

    print("Best train_acc: {}".format(history.history['acc'][best_epoch_idx]))
    print("Best train_loss: {}".format(history.history['loss'][best_epoch_idx]))
    print("Best val_acc: {}".format(history.history['val_acc'][best_epoch_idx]))
    print("Best val_loss: {}".format(history.history['val_loss'][best_epoch_idx]))

    #####################################################################
    #                       Make predicitions
    #####################################################################

    print("Calculating predictions...")

    start = time.time()

    predictions = []
    paths = test.path.tolist()

    for path in paths:
        specgrams = []
        specgram = None

        if model_name in ["vgg16", "inception_v3", "resnet50_v2"]:
            _, _, specgram = get_log_spectrogram(path, num_channels=3)
        else:
            _, _, specgram = get_log_spectrogram(path, num_channels=1)

        specgrams.append(specgram)
        pred = model.predict(np.array(specgrams))
        # print(pred)
        # argmax = np.argmax(pred, axis=1)
        # print("Argmax: {}".format(argmax))

        # label = labelbinarizer.classes_[argmax]
        # print(label)
        predictions.append(pred)

    end = time.time()
    print("Calculating predicitions time: {}".format(end - start))

    print("Transforming predicitions to labels and writing csv...")

    labels = [labelbinarizer.classes_[np.argmax(p, axis=1)][0] for p in predictions]
    test['labels'] = labels
    test.path = test.path.apply(lambda x: str(x).split('/')[-1])
    submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
    submission.to_csv('simple-keras-model-with-data-generator_submission_new.csv', index=False)