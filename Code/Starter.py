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
