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


import os
import tensorflow as tf
from NFLossFunctions import STFT_loss, centLoss
from GetData import get_data_single, get_data_single_2
from Models import create_model_NF
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
import pickle
import random
import numpy as np


def train(data_dir, epochs, **kwargs):

    """
      :param data_dir: the directory in which dataset are stored [string]
      :param b_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param phase: the training phase [string]
      :param steps: number of timesteps to generate per iteration [int]
      :param harmonics: number of partial to generate [int]
      :param scenario: which scenario to consider [string]
      :param epochs: the number of epochs [int]
    """
    
    b_size = kwargs.get('b_size', 1)
    learning_rate = kwargs.get('learning_rate', 0.00001)

    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')

    inference = kwargs.get('inference', False)
    phase = kwargs.get('phase', 'B')
    steps = kwargs.get('steps', 240)
    harmonics = kwargs.get('harmonics', 24)
    scenario = kwargs.get('scenario', '1')
    
    #tf.keras.backend.set_floatx('float64')

    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    # define the Adam optimizer with the initial learning rate, training steps
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

    # depending on the phase different losses and layers are considered for the training
    if phase == 'B':
        w = [0., 1., 0.]
        train_B = True
        train_amps = False
    elif phase == 'A':
        w = [1., 0., 1.]
        train_B = False
        train_amps = True
    else:
        w = [1., 1., 1.]
        train_B = True
        train_amps = True

    # create the model
    model = create_model_NF(poly=1, num_steps=steps, harmonics=harmonics, train_B=train_B, train_amps=train_amps)
    
    # define the losses and their weights
    lossesName = ["tf.__operators__.getitem_3", "tf.__operators__.getitem_4", "tf.math.abs_2"]
    losses = {
        lossesName[0]: STFT_loss(m=[512, 1024, 2048]),
        lossesName[1]: centLoss(delta=1),
        lossesName[2]: "mse",
    }
    lossWeights = {lossesName[0]: w[0], lossesName[1]: w[1], lossesName[2]: w[2]}

    # compile the model
    model.compile(loss=losses, loss_weights=lossWeights, metrics=['mse'], optimizer=opt)

    # define callbacks: where to store the weights
    callbacks = []
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2)
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest, scheduler]

    if scenario == '1': # unseen key
    
        f, v, y, f0, amp, ind, f_val, v_val, y_val, f0_val, amp_val, ind_val, f_test, v_test, y_test, f0_test, amp_test, ind_test = get_data_single(data_dir=data_dir, step=steps)
    
    elif scenario == '2': # unseen velocity
        f, v, y, f0, amp, ind, f_val, v_val, y_val, f0_val, amp_val, ind_val, f_test, v_test, y_test, f0_test, amp_test, ind_test = get_data_single_2(data_dir=data_dir, step=steps)

     # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        results = model.fit([f, v, ind],
                  y={lossesName[0]: y, lossesName[1]: f0, lossesName[2]: amp},
                  batch_size=1,
                  epochs=epochs, verbose=0,
                  validation_data=([f_val, v_val, ind_val],
                                {lossesName[0]: y_val, lossesName[1]: f0_val,
                                lossesName[2]: amp_val}),
                callbacks=callbacks)
        
        # write and save results
        writeResults(None, results, b_size, learning_rate, model_save_dir, save_folder, 1)

        # plot the training and validation loss for all the training
        loss_training = results.history['loss']
        loss_val = results.history['val_loss']
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, '')

        print("Training done")

    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found, there is something wrong
        print("Something is wrong.")

    # compute test loss
    # reset the states before predicting
    model.reset_states()
    test_loss = model.evaluate(x=[f_test, v_test, ind_test],
                                   y=[{lossesName[0]: y_test, lossesName[1]: f0_test, lossesName[2]: amp_test}],
                                   batch_size=1, verbose=0, return_dict=True)
    results_ = {'Test_Loss': test_loss}
    
    # write and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results_, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    # reset the states before predicting
    model.reset_states()
    predictions = model.predict([f_test, v_test, ind_test], batch_size=1, verbose=0)
    predictions = predictions[0]
    
    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, y_test, model_save_dir, save_folder, 24000, steps, scenario)

    return 42
