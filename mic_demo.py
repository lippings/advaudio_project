import pyaudio
import struct
import matplotlib.pyplot as plt
import yaml
from tkinter import TclError
from keras.models import load_model
import os
import librosa
import numpy as np
from time import time
import tensorflow as tf


def eval_feat(data, window_size, window_stride, window, normalize, max_len=101, sr=16000):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length,
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


def load_settings_file(file_path):
    with open(file_path) as f:
        return yaml.load(f)


def load_input_settings(file_path):
    settings = load_settings_file(file_path)["input_format"]

    return settings["chunk"],\
           settings["format"],\
           settings["channels"],\
           settings["rate"]


def main():
    settings_file = 'settings/mic_demo.yaml'

    model = load_model('CNNmodel.h5')

    window_size = .02
    window_stride = .01
    window_type = 'hamming'
    normalize = True
    max_len = 101

    classes = os.listdir('./dataset/part1/val/')

    chunk, format, channels, rate = load_input_settings(settings_file)
    format = eval(format)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        output=True,
        frames_per_buffer=chunk
    )

    fig, ax = plt.subplots()

    frames = [np.random.randn(chunk)]*4

    line, = ax.plot(np.random.randn(4*chunk))
    plt.show(block=False)
    plt.ylim([-1.5, 1.5])
    fig.canvas.draw()

    while True:
        try:
            data = stream.read(chunk)
            data_float = np.array(struct.unpack(str(2 * chunk) + 'B', data), dtype='b')[::2]
            data_float = np.float32(data_float / (2**7))
            del frames[0]
            frames.append(data_float)

            sec_signal = np.concatenate(frames)
            line.set_ydata(sec_signal)

            in_feat = eval_feat(sec_signal, window_size, window_stride, window_type, normalize, max_len=max_len)
            in_feat = np.swapaxes(in_feat, 0, 2)
            in_feat = np.array([in_feat])

            pred = model.predict(in_feat)
            ind = np.argmax(pred)

            pred_class = classes[ind]
            # plt.title(pred_class)
            if pred_class != "silence":
                print(pred_class)

            ax.draw_artist(ax.patch)
            ax.draw_artist(line)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except TclError:
            print("Stream stopped!")
            break


if __name__ == "__main__":
    main()
