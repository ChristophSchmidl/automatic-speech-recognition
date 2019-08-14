import matplotlib.pyplot as plt
from preprocessing import *

def plot_9_log_spectrograms(data):
    # Show me 9 samples using log spectrograms
    fig = plt.figure(figsize=(10, 10))

    for idx, item in enumerate(data[:9]):
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


def plot_9_spectrograms(data):
    # Show me 9 samples using spectrograms
    fig = plt.figure(figsize=(10, 10))

    for idx, item in enumerate(data[:9]):
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



def plot_single_spectrogram(log_spectrogram, label):
    fig = plt.figure()
    plt.title(label)
    plt.imshow(log_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    plt.show(block=False)


def plot_single_log_spectrogram(spectrogram, label):
    fig = plt.figure()
    plt.title(label)
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')
    plt.show(block=False)




def visualize(train_full, train_reduced):
    #####################################################################
    #      Visualize first sample from each training class
    #####################################################################

    # Give me the first occurence of every class as list: every item has word, path
    sample_of_each_class = train_full.groupby('word', as_index=False).first().values

    plot_9_spectrograms(sample_of_each_class)
    plot_9_log_spectrograms(sample_of_each_class)

    #####################################################################
    #      Visualize random sample from full training set
    #####################################################################

    train_full_arr = train_full.values
    X_full = train_full_arr[:, 0]
    y_full = train_full_arr[:, 1]

    # print("X_full:")
    # print(X_full)

    # print("y_full:")
    # print(y_full)

    # Give me a random spectrogram and log_spectrogram

    rand_input, rand_label = get_random_input_and_label(X_full, y_full)

    print("Random input: {}".format(rand_input))
    print("Random label: {}".format(rand_label))

    _, _, random_spectrogram = get_spectrogram(rand_input)
    _, _, random_log_spectrogram = get_log_spectrogram(rand_input)

    plot_single_spectrogram(random_spectrogram, "Full train spectrogram: " + rand_label)
    plot_single_log_spectrogram(random_log_spectrogram, "Full train log spectrogram: " + rand_label)

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

    plot_single_spectrogram(random_spectrogram, "Reduced train spectrogram: " + rand_label)
    plot_single_log_spectrogram(random_log_spectrogram, "Reduced train log spectrogram: " + rand_label)

    print("Dimensions of spectrogram: {}".format(random_spectrogram.shape))
    print("Dimensions of log spectrogram: {}".format(random_log_spectrogram.shape))

    plt.show(block=True)