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
from scipy import fft
import glob

def plotting(p, y, model_save_dir, save_folder, step, scenario):
    """
    Plot the rendered results
      :param p: the model's prediction  [array of floats]
      :param y: the target [array of floats]
      :param model_save_dir: the director where the models are saved [string]
      :param save_folder: the director where the model is saved [string]
      :param step: the number of steps per iteration [int]
      :param scenario: the name of the file [string]
    """
    
    dir = model_save_dir + '/' + save_folder + '/'

    l = 57600
    fs = 24000
    N_stft = 2048
    N_fft = fs*2
    frame = 2400

    if scenario == '1':
        max = 5
    elif scenario == '2':
        max = 23

    for i in range(max):
        tar = y[i * l: (i + 1) * l]
        pred = p[i * l: (i + 1) * l]

        t = np.arange(0, len(pred) / (frame // 4))

        #FFT
        FFT_t = np.abs(fft.fftshift(fft.fft(tar, n=N_fft))[N_fft // 2:])
        FFT_p = np.abs(fft.fftshift(fft.fft(pred, n=N_fft))[N_fft // 2:])
        freqs = fft.fftshift(fft.fftfreq(N_fft) * fs)
        freqs = freqs[N_fft // 2:]


        fig, ax = plt.subplots(1, 1)
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_t)/N_fft), color='deepskyblue', label='Target')
        ax.semilogx(freqs, 20 * np.log10(np.abs(FFT_p)/N_fft), '--', color='red', label='Prediction')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xticks([300, 1000, 3000, 6000])
        ax.set_xticklabels(['300', '1000', '3000', '6000'])
        ax.axis(xmin=100, xmax=8000)
        ax.axis(ymin=-180, ymax=-40)
        ax.legend(loc='upper right')
        fig.savefig(dir + str(i) + 'FFT.pdf', format='pdf')
        plt.close('all')

        #STFT
        D = librosa.stft(tar, n_fft=N_stft)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
        ax[0].set_title('STFT Magnitude (Top: target, Bottom: prediction)')
        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_xlabel('Time [sec]')
        ax[0].label_outer()

        D = librosa.stft(pred, n_fft=N_stft)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[1])
        ax[1].set_ylabel('Frequency [Hz]')
        ax[1].set_xlabel('Time [sec]')
        fig.savefig(dir + str(i) + 'STFT.pdf', format='pdf')
        plt.close('all')
        
        
        # Time
        fig, ax = plt.subplots(1, 1)
        pr = librosa.feature.rms(pred, frame_length=frame, hop_length=frame//4, center=True)[0][:-1]
        rr = librosa.feature.rms(tar, frame_length=frame, hop_length=frame//4, center=True)[0][:-1]

        if len(t) > len(pr):
            t = t[:len(pr)]

        ax.plot(t / fs * 1000, pr, 'b', label='Prediction')
        ax.plot(t / fs * 1000, rr, 'g--', label='Target')
        ax.label_outer()
        ax.legend(loc='upper right')
        ax.axis(ymin=0, ymax=np.max(rr)+0.01)
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (ms)')
        fig.savefig(dir + str(i) + '_rms_plot.pdf', format='pdf')
        plt.close('all')



            
