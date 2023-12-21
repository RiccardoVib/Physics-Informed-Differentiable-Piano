from Training import trainED

DATA_DIR = '../../Files' #### Dataset folder
MODEL_SAVE_DIR = '../../TrainedModels' #### Models folder
MODEL_NAME = '240_model_2_relu' #### Model name

INFERENCE = True#False
EPOCHS = 10
PHASE = 'B'
HARMONICS = 24
STEPS = 24
LR = 1e-6
SCENARIO = '2'


trainED(data_dir=DATA_DIR,
        model_save_dir=MODEL_SAVE_DIR,
        save_folder=MODEL_NAME,
        learning_rate=LR,
        epochs=EPOCHS,
        steps=STEPS,
        harmonics=HARMONICS,
        phase='B',
        scenario=SCENARIO,
        inference=INFERENCE)
