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
from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, UpSampling2D, Concatenate, Add, Lambda, Conv1D
from tensorflow.keras.models import Model
import math as m
import numpy as np

def create_model_NF(num_steps=240, harmonics=32, units=64, train_B=True, train_amps=True, batch_size=1):
    """
    piano model
          :param num_steps: number of steps to generate per iteration
          :param harmonics: number of partials to generate
          :param units: number of units
          :param train_B: if training layer computing B
          :param train_amps: if training layer computing amplitudes' partials
          :param batch_size: the batch size

    """

    # [input is vector Mx1x2, 1 row is the vel, the other the freq0]
    type = tf.float32

    # prepare parameters needed for later
    max_ = tf.constant(493.8833, dtype=type)
    twopi = tf.constant(2 * m.pi, dtype=type)
    Fs = 24000
    n = tf.expand_dims(tf.expand_dims(tf.linspace(1, harmonics, harmonics), axis=0), axis=0)
    n = tf.cast(n, dtype=type)
    ones = tf.expand_dims(tf.ones((1, harmonics)), axis=0)

    # defining inputs
    freq_inputs = Input(shape=(1, 1), batch_size=batch_size, name='freq_input')
    vel_inputs = Input(shape=(1, 2), batch_size=batch_size, name='vel_inputs')
    index_inputs = Input(shape=(1), batch_size=batch_size, name='k_inputs')

    # normilize input frequency
    f_n = tf.divide(freq_inputs, max_)

    # defining network predicting B factor
    B = Dense(32, name='B', trainable=train_B, dtype=type)(f_n)
    B = Dense(1, activation='relu', name='B2', trainable=train_B, dtype=type)(B)

    # computing the partials distribution given the B factor
    n2 = tf.pow(n, 2)
    B = tf.abs(B)  # 1x1x1
    Bn = Multiply()([n2, B])  # 1xHx1
    Bn = Add()([ones, Bn])
    Bn = tf.sqrt(Bn)

    final_n = Multiply()([n, Bn[:, 0, :]])
    freqs_inputs = Multiply()([final_n, freq_inputs])  # Bx1xH    

    # generating sine waves according to partials frequencies
    f = tf.divide(freqs_inputs, Fs, name='divide')  # Bx1xH
    x = tf.multiply(twopi, f, name='mul')  # Bx1xH
    t = tf.ones((1, D, num_steps, harmonics))  # Bx1xTxH
    t = tf.constant(t * np.arange(num_steps).reshape(1, 1, num_steps, 1), dtype=type)

    k = tf.expand_dims(index_inputs, axis=-1)
    k = tf.expand_dims(k, axis=-1)
    k = tf.multiply(k, num_steps)
    t = t + k
    x = tf.expand_dims(x, axis=2)  # BxDx1xH
    xt = tf.multiply(x, t, name='mul2')  # BxDxTxH
    sines = tf.sin(xt, name='sins')  # BxDxTxH

    # concatenate frequency and velocity inputs
    all_inp = Concatenate(axis=-1)([f_n, vel_inputs[:, :, :2]])

    # defining network predicting partials amplitudes
    ampsb1 = LSTM(1, name='decay_amp_sins1', activation='sigmoid', trainable=train_amps, dtype=type)(all_inp)
    ampsb3 = LSTM(1, name='decay_amp_sins3', trainable=train_amps, dtype=type)(all_inp)

    amps_2 = LSTM(harmonics, return_sequences=True, name='coefficient_amp_sins', trainable=train_amps, dtype=type)(all_inp)
    amps_2 = Dense(harmonics, name='coefficient_amp_sins2', activation='sigmoid', trainable=train_amps, dtype=type)(amps_2)
    amps_2 = tf.divide(amps_2, harmonics)
    amps_2 = tf.expand_dims(amps_2, axis=2)  # Bx1x1xH

    # compute the decay rates
    f2 = tf.pow(f, 2)
    f2 = tf.multiply(twopi, f2, name='mul1')  # Bx1xH
    decay_rate = tf.multiply(f2, ampsb3, name='mul2')  # Bx1xH #####
    decay_rate = tf.add(ampsb1, decay_rate, name='mul2')
    decay_rate = tf.abs(decay_rate, name='abs1')  # Bx1xH
    decay_rate = tf.expand_dims(decay_rate, axis=2)  # Bx1x1xH
    t_s = tf.divide(t, Fs)
    decay_rate = tf.multiply(decay_rate, t_s, name='mul3')
    decay = tf.math.exp(-decay_rate, name='exp1')
    decay = tf.multiply(decay, amps_2, name='mul4')

    # defining network predicting the detuned frequency for the beatings
    all_inp_b = Concatenate(axis=-1)([f_n, vel_inputs[:, :, :1]])
    beatings = Dense(1, name='beatings', activation='tanh', trainable=train_amps, dtype=type)(all_inp_b)

    # compute the beatings
    freqs_beatings = tf.add(freq_inputs, beatings)
    freqs_beatings = Multiply()([final_n_c, freqs_beatings])
    f_b = tf.divide(freqs_beatings, Fs, name='divide')  # BxDxH
    x_b = tf.multiply(twopi, f_b, name='mul')  # BxDxH
    x_b = tf.expand_dims(x_b, axis=2)  # BxDx1xH
    xt_b = tf.multiply(x_b, t, name='mul2')  # BxDxTxH
    sines_beatings = tf.sin(xt_b, name='sins')# BxDxTxH
    sines_final = tf.add(sines, sines_beatings)
    all_harmonics = tf.multiply(decay, sines_final, name='mulf')

    # compute the longitudinal waves (double frequency)
    xt_l = tf.multiply(2.0, xt)
    sines_l = tf.sin(xt_l, name='sins_l')  # BxDxTxH
    decay_rate_l = tf.divide(decay_rate, 0.5)
    decay_l = tf.math.exp(-decay_rate_l, name='exp9')
   
    decay_l = tf.multiply(decay_l, amps_2, name='mul410')
    decay_sines_l = tf.multiply(decay_l, sines_l, name='mul50')  # BxDxTxH
    all_harmonics = tf.add(all_harmonics, decay_sines_l)

    # sum all partials
    all_harmonics = tf.math.reduce_sum(all_harmonics, axis=-1, name='reduce_sum1') #sum harmonics

    all_harmonics = tf.math.reduce_sum(all_harmonics, axis=1, name='reduce_sum2')  # sum notes

    # compute the rms
    rms = tf.abs(tf.reduce_mean(tf.square(all_harmonics), axis=-1))
    model = Model([freq_inputs, vel_inputs, index_inputs], [all_harmonics[0], freqs_inputs[0, :, 0], rms])#, all_noise[0]])
    model.summary()
    return model
