import os
import tensorflow as tf
from NFLossFunctions import STFT_loss, centLoss
from GetData import get_data_single, get_data_single_2
from Models import create_model_NF
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
import pickle
import random
import numpy as np


def trainED(data_dir, epochs, **kwargs):
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
    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Learning rate scheduler
    min_learning_rate = learning_rate

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

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


    model = create_model_NF(poly=1, num_steps=steps, harmonics=harmonics, train_B=train_B, train_amps=train_amps)
    lossesName = ["tf.__operators__.getitem_3", "tf.__operators__.getitem_4", "tf.math.abs_2"]
    losses = {
        lossesName[0]: STFT_loss(m=[512, 1024, 2048]),
        lossesName[1]: centLoss(delta=1),
        lossesName[2]: "mse",
    }
    lossWeights = {lossesName[0]: w[0], lossesName[1]: w[1], lossesName[2]: w[2]}

    model.compile(loss=losses, loss_weights=lossWeights, metrics=['mse'], optimizer=opt)
    callbacks = []
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2)
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, '')
    callbacks += [ckpt_callback, ckpt_callback_latest, scheduler]

    if scenario == '1':
    
        f, v, y, f0, amp, ind, f_val, v_val, y_val, f0_val, amp_val, ind_val, f_test, v_test, y_test, f0_test, amp_test, ind_test = get_data_single(data_dir=data_dir, step=steps)
    
    elif scenario == '2':
        f, v, y, f0, amp, ind, f_val, v_val, y_val, f0_val, amp_val, ind_val, f_test, v_test, y_test, f0_test, amp_test, ind_test = get_data_single_2(data_dir=data_dir, step=steps)
        
    if not inference:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

        results = model.fit([f, v, ind],
                  y={lossesName[0]: y, lossesName[1]: f0, lossesName[2]: amp},
                  batch_size=1,
                  epochs=epochs, verbose=0,
                  validation_data=([f_val, v_val, ind_val],
                                {lossesName[0]: y_val, lossesName[1]: f0_val,
                                lossesName[2]: amp_val}),
                callbacks=callbacks)

        writeResults(None, results, b_size, learning_rate, model_save_dir, save_folder, 1)
        loss_training = results.history['loss']
        loss_val = results.history['val_loss']
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, '')

        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)


        test_loss = model.evaluate(x=[f_test, v_test, ind_test],
                                   y=[{lossesName[0]: y_test, lossesName[1]: f0_test, lossesName[2]: amp_test}],
                                   batch_size=1, verbose=0, return_dict=True)
        results_ = {'Test_Loss': test_loss}
        print(results_)

        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results_.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results_, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

        print("Training done")

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best)

    predictions = model.predict([f_test, v_test, ind_test], batch_size=1, verbose=0)
    predictions = predictions[0]
    predictWaves(predictions, y_test, model_save_dir, save_folder, 24000, steps, scenario)

    return 42
