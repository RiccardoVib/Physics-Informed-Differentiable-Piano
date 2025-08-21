# Physics-informed Differentiable Method for Piano Modeling

This code repository for the article _Physics-informed Differentiable Method for Piano Modeling_, Frontiers in Signal Processing 3 (2024).

This repository contains all the necessary utilities to use our architecture. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/Physics-Informed-Differentiable-Piano_pages/)

### Folder Structure

```
./src
├── AudioExamples
├── Code
└── Weights
    ├── Scenario A
    └── Scenario B
```

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

### Dataset:
Datsets is available at the following link:
[Piano Recordings Single Notes](https://doi.org/10.34740/kaggle/dsv/6285806)

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./Code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=" ")
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=8)
* --steps - Number of steps to generate [int] (default=240)
* --harmonics - Number of harmonics to synthetize [int] (default=24)
* --scenario = Scenario to evaluate: unseen key (1),  unseen velocity (2) [ [int] ] (default='1')
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --datasets 'pianoUpright' --steps 240 --harmonics 24 --scenario '1' --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --datasets 'pianoUpright' --steps 240 --harmonics 24 --scenario '1' --only_inference True
```

# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```
@article{simionato2024physics,
  title={Physics-informed differentiable method for piano modeling},
  author={Simionato, R. and Fasciani, S. and Holm, S.},
  journal={Frontiers in Signal Processing},
  volume={3},
  pages={1276748},
  year={2024},
  publisher={Frontiers Media SA}
}
```