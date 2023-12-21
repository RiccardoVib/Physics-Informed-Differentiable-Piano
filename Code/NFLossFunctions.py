import tensorflow as tf
import numpy as np
from statistics import mean
from tensorflow.keras import backend as K
from scipy import signal


class centLoss(tf.keras.losses.Loss):
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