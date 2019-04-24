from keras.models import load_model
import os
import numpy as np
import librosa
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt


def eval_feat(y, window_size, window_stride, window, normalize, max_len=101, sr=16000):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    # spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = np.mean(np.ravel(spect))
        std = np.std(np.ravel(spect))
        if std != 0:
            spect = spect - mean
            spect = spect / std

    return spect


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101,
                 augment=False, allow_speedandpitch=False, allow_pitch=False,
                 allow_speed=False, allow_dyn=False, allow_noise=False,
                allow_timeshift=False ):
    y, sr = librosa.load(path, sr=None)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    #spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = np.mean(np.ravel(spect))
        std = np.std(np.ravel(spect))
        if std != 0:
            spect = spect -mean
            spect = spect / std

    return spect


tf.logging.set_verbosity(tf.logging.ERROR)
##-------------MUUTA NÄITÄ-------------------
dir = "dataset/oma"
classes = os.listdir('./dataset/part1/val/')

ifile = "up.wav"
ifile = os.path.join(dir, ifile)
##-------------MUUTA NÄITÄ-------------------

model = load_model("CNNmodel.h5")


window_size = .02
window_stride = .01
window_type = 'hamming'
normalize = True
max_len = 101
#
# in_feat = spect_loader(ifile, window_size, window_stride, window_type, normalize, max_len=max_len)
# in_feat = np.swapaxes(in_feat, 0, 2)
# in_feat = np.array([in_feat])

y, sr = librosa.load(ifile, sr=None)

iters = 10
start = time()
for _ in range(iters):
    in_feat = eval_feat(y, window_size, window_stride, window_type, normalize, max_len=max_len)
    in_feat = np.swapaxes(in_feat, 0, 2)
    in_feat = np.array([in_feat])

    pred = model.predict(in_feat)
    ind = np.argmax(pred)

    print("Predicted class: {}".format(
        classes[ind]
    ))

plt.matshow(np.reshape(in_feat, [101, 161]))
plt.show()

print("Time per pass: {:.2f}".format(
    (time()-start)/iters
))
