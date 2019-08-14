from pathlib import Path
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    words = df.word.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown too.
    # df.loc[df.word.isin(silence), 'word'] = 'unknown'
    # df.loc[df.word.isin(unknown), 'word'] = 'unknown'

    # I want to have 12 labels: train_words, unknown and silence
    df.loc[df.word.isin(silence), 'word'] = 'silence'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'

    return df


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def get_spectrogram(audio_path, window_size=20, step_size=10, eps=1e-10, num_channels=1):
    (sample_rate, sig) = wavfile.read(audio_path)

    #print("Samplerate of wav file: {}".format(sample_rate))

    if sig.size < sample_rate:
        sig = np.pad(sig, (sample_rate - sig.size, 0), mode='constant')
    else:
        sig = sig[0:sample_rate]

    # f = array of sample frequencies
    # t = array of segment times
    # Sxx = Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
    f, t, spectrogram = signal.spectrogram(sig, nperseg=256, noverlap=128)
    spectrogram = spectrogram.reshape(99, 161, -1)
    spectrogram = (np.dstack([spectrogram] * num_channels)).reshape(99, 161, -1)  # Make it fit to VGG16 and 3 channels

    return f, t, spectrogram


def get_log_spectrogram(audio_path, window_size=20, step_size=10, eps=1e-10, num_channels=1):
    (sample_rate, sig) = wavfile.read(audio_path)

    #print("Samplerate of wav file: {}".format(sample_rate))

    if sig.size < 16000:
        sig = np.pad(sig, (sample_rate - sig.size, 0), mode='constant')
    else:
        sig = sig[0:sample_rate]

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    # f = array of sample frequencies
    # t = array of segment times
    # Sxx = Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
    f, t, Sxx = signal.spectrogram(sig,
                                   fs=sample_rate,
                                   window='hann',
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   detrend=False)
    log_spectrogram = np.log(Sxx.T.astype(np.float32) + eps)
    log_spectrogram = log_spectrogram.reshape(99, 161, -1)
    log_spectrogram = (np.dstack([log_spectrogram] * num_channels)).reshape(99, 161, -1)  # Make it fit to VGG16 and 3 channels

    return f, t, log_spectrogram



def get_random_index(arr):
    idx = np.random.randint(0, arr.shape[0])
    return idx


def get_random_input_and_label(X, y):
    # choose batch_size random images / labels from the data
    idx = get_random_index(X)
    audio_path = X[idx]
    label = y[idx]

    return audio_path, label


def batch_generator(X, y, batch_size=16, num_channels=1):
    '''
    Return a random image from X, y
    '''

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        audio_paths = X[idx]
        labels = y[idx]

        # VGG16 works with 3 channels but I only got one. Therefore, evil hacking ahead...

        specgrams = [get_log_spectrogram(x, num_channels=num_channels)[2] for x in audio_paths]

        #print(specgram)

        #print(specgrams.shape)

        yield np.concatenate([specgrams]), labels



