# Copyright (C) 2023 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# Simionato, Riccardo, Stefano Fasciani, and Sverre Holm. "Physics-informed differentiable method for piano modeling." Frontiers in Signal Processing 3 (2024): 1276748.


import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile
import librosa.display
from scipy import fft, signal
from Plotting import plotting


def get_batches(x, b_size=1, shuffle=True, seed=99):
    np.random.seed(seed)
    indxs = np.arange(tf.shape(x)[0])
    if shuffle:
        np.random.shuffle(indxs)

    def divide_chunks(l, n):
        # looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    x_b, y_b, z_b = [], [], []
    x1_b = []
    indxs = divide_chunks(indxs, b_size)

    for indx_batch in indxs:
        # if len(indx_batch) != b_size:
        #     continue
        x_b.append(x[indx_batch])

    return x_b

def get_indexed_shuffled(x, seed=99):
    np.random.seed(seed)
    indxs = np.arange(tf.shape(x)[0])
    np.random.shuffle(indxs)

    def divide_chunks(l, n):
        # looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    indxs = divide_chunks(indxs, tf.shape(x)[0])


    return indxs

def writeResults(test_loss, results, b_size, learning_rate, model_save_dir, save_folder,
                 index):
    results = {
        'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(predictions, y, model_save_dir, save_folder, step):
    
    if step==24:
        step=step*10
        
    l = step * 240
    fs = 24000
    N_stft = 2048
    N_fft = fs*2

    for i in range(6):
        tar = y[i * l: (i + 1) * l]
        pred = predictions[i * l: (i + 1) * l]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(predictions, label='pred')
        # ax.plot(x, label='inp')
        # ax.plot(y, label='tar')
        display.waveshow(tar, sr=fs, ax=ax, label='Target', alpha=0.9)
        display.waveshow(pred, sr=fs, ax=ax, label='Prediction', alpha=0.7)
        # ax.label_outer()
        ax.legend(loc='upper right')
        fig.savefig(model_save_dir + '/' +save_folder + '/' + str(i) + 'plot')
        plt.close('all')
        
        
        #FFT
        FFT_t = np.abs(fft.fftshift(fft.fft(tar, n=N_fft))[N_fft // 2:])
        FFT_p = np.abs(fft.fftshift(fft.fft(pred, n=N_fft))[N_fft // 2:])
        freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
        freqs = freqs[N_fft // 2:]

        fig, ax = plt.subplots(1, 1)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)), label='Target',)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)), label='Prediction')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.axis(xmin=20, xmax=22050)

        ax.legend(loc='upper right')
        fig.savefig(model_save_dir  + '/' + save_folder + '/' + str(i) + 'FFT.pdf', format='pdf')
        plt.close('all')

        #STFT
        D = librosa.stft(tar, n_fft=N_stft)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        #ax[0].pcolormesh(t, f, np.abs(Zxx), vmin=np.min(np.abs(Zxx)), vmax=np.max(np.abs(Zxx)), shading='gouraud')
        librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
        ax[0].set_title('STFT Magnitude (Top: target, Bottom: prediction)')
        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_xlabel('Time [sec]')
        ax[0].label_outer()

        #f, t, Zxx = signal.stft(_p, fs, nperseg=1000)
        D = librosa.stft(pred, n_fft=N_stft)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[1])
        #ax[1].pcolormesh(t, f, np.abs(Zxx), vmin=np.min(np.abs(Zxx)), vmax=np.max(np.abs(Zxx)), shading='gouraud')
        ax[1].set_ylabel('Frequency [Hz]')
        ax[1].set_xlabel('Time [sec]')
        fig.savefig(model_save_dir  + '/' + save_folder + '/' + str(i) + 'STFT.pdf', format='pdf')
        plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder, name):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + name + 'loss.png')
    plt.close('all')


def predictWaves(predictions, y_test, model_save_dir, save_folder, fs, step, scenario):
    pred_name = '_pred.wav'
    tar_name = '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    #plotResult(predictions.reshape(-1), y_test.reshape(-1), model_save_dir, save_folder, step)

    plotting(predictions.reshape(-1), y_test.reshape(-1), model_save_dir, save_folder, step, scenario)


def checkpoints(model_save_dir, save_folder, name):
    ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(
        os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'latest', 'latest.ckpt'))
    ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'best'))
    ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, name, 'Checkpoints', 'latest'))

    if not os.path.exists(os.path.dirname(ckpt_dir)):
        os.makedirs(os.path.dirname(ckpt_dir))
    if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
        os.makedirs(os.path.dirname(ckpt_dir_latest))

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1)
    ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False, save_weights_only=True,
                                                              verbose=1)

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest
