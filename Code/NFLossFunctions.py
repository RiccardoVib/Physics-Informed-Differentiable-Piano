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


import tensorflow as tf
import numpy as np
from statistics import mean
from tensorflow.keras import backend as K
from scipy import signal


class centLoss(tf.keras.losses.Loss):
    """ cent loss """
    def __init__(self, delta=0., name="Cent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.keras.metrics.mean_squared_error(tf.experimental.numpy.log2(y_true+self.delta), tf.experimental.numpy.log2(y_pred+self.delta))

        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}


class STFT_loss(tf.keras.losses.Loss):
    """ multi-STFT loss """
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):

        loss = 0
        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            Y_true = K.pow(K.abs(Y_true), 2)
            Y_pred = K.pow(K.abs(Y_pred), 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)

        return loss

    def get_config(self):
        config = {
            'm': self.m
        }
        base_config = super().get_config()
        return {**base_config, **config}

class STFT_and_imag_loss(tf.keras.losses.Loss):
    """ multi-STFT loss + image loss """

    def __init__(self, m=[32, 64, 128, 256, 512, 1024], name="STFT_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m

    def call(self, y_true, y_pred):

        loss = 0
        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred = K.abs(tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            Y_true = K.pow(K.abs(Y_true), 2)
            Y_pred = K.pow(K.abs(Y_pred), 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            Y_true_i = tf.math.imag(tf.signal.stft(y_true, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))
            Y_pred_i = tf.math.imag(tf.signal.stft(y_pred, fft_length=self.m[i], frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True))

            loss += tf.norm((Y_true_i - Y_pred_i), ord=1)
            loss += tf.norm((l_true - l_pred), ord=1) + tf.norm((Y_true - Y_pred), ord=1)

        return loss
