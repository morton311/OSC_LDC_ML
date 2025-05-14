# nohup /home/cmorton/Desktop/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows/.venv/bin/python3 -u /home/HDD/cmorton/OSC_LDC_ML/ldc_predict.py > pred.log &

print('Loading libraries...')
import h5py
import pickle
import os
import gc
import json

import numpy as np
import math
import torch 
import torch.nn as nn
from functools import partial
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import matplotlib.pyplot as plt
import time
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime

from lib.dls import gfem_2d, gfem_recon  # Explicitly import required functions
from lib.dls_funcs import gfem_2d, gfem_recon  # Explicitly import required functions
from lib.eval import * # Imports analysis functions
from lib.transformer import TransformerEncoderModel, make_Sequence, train_model, normalize_data, denormalize_data, predict

print('Libraries loaded.')

# Load the combined configuration file
config_path = "configs/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
    # Print the keys and nested keys in the config file
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
                # load the sub_key as a variable
                globals()[sub_key] = sub_value


# Access parameters from the JSON file
data_name = config["data"]["data_name"]
patch_size = config["data"]["patch_size"]
num_modes = config["data"]["num_modes"]
num_train = config["train"]["num_train"]
num_test = config["train"]["num_test"]
time_lag = config["transformer"]["time_lag"]
d_model = config["transformer"]["d_model"]
nhead = config["transformer"]["nhead"]
num_layers = config["transformer"]["num_layers"]
ram_available = config["misc"]["ram_available"]

## ====================================== Directory Handle ==========================================
# Make directories for saving figures and animations
fig_dir = 'figs/' + data_name + '/'
latent_id = 'm' + str(num_modes) + 'p' + str(patch_size) + '/'
anim_dir = 'anim/' + data_name + '/'

os.makedirs('figs', exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(fig_dir + latent_id, exist_ok=True)
os.makedirs('anim', exist_ok=True)
os.makedirs(anim_dir, exist_ok=True)
os.makedirs(anim_dir + latent_id, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Define paths for saving model and predictions
model_dir = f'models/{data_name}/{latent_id}'
predictions_dir = f'predictions/{data_name}/{latent_id}'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

model_save_path = os.path.join(model_dir, 'model.pth')
normalization_params_save_path = os.path.join(model_dir, 'normalization_params.npz')

latent_name = 'dls_' + data_name + '_m' + str(num_modes) + '_p' + str(patch_size)
latent_file = 'latent/' + latent_name + '.h5'
latent_config_file = 'latent/' + latent_name + '_config.pkl'
latent_metrics = 'latent/' + latent_name + '_metrics.txt'



# Load latent variables
print('Loading latent variables...')
with h5py.File(latent_file, 'r') as f:
    dof_u = f['dof_u'][:time_lag]
    dof_v = f['dof_v'][:time_lag]

# Load normalization parameters
print('Loading normalization parameters...')
normalization_params = np.load(normalization_params_save_path)
mean = normalization_params['mean']
std = normalization_params['std']
print('Normalization parameters loaded.') 

input = normalize_data(np.concatenate((dof_u , dof_v), axis=1), mean, std)
print(f'Input shape: {input.shape}')
print('Latent variables loaded and normalized.')

input_dim = input.shape[1]


## Load model
print('Loading model...')
model_name = f'{data_name}_best_model'
model_path = f'{model_dir}/{model_name}.pth'
model = TransformerEncoderModel(
    time_lag=time_lag,
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
).to(device)

# Load the model state dictionary
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()  # Set the model to evaluation mode
print('Model loaded.')


# Predict to num_predictions
num_predictions = 2 * num_train - time_lag 
print(f'Predicting {num_predictions} time steps...')
predictions = predict(model, input, time_lag, num_predictions, device)
print('Predictions completed.')

# denormalize predictions
predictions = denormalize_data(predictions, mean, std)
# Save predictions
with h5py.File(os.path.join(predictions_dir, f'predictions_num{num_predictions}.h5'), 'w') as hf:
    hf.create_dataset('predictions', data=predictions)
pred_path = os.path.join(predictions_dir, f'predictions_num{num_predictions}.h5')
print(f'Predictions saved to {pred_path}.')

## Pred on unseen data
print('Loading unseen data...')
with h5py.File(latent_file, 'r') as f:
    dof_u = f['dof_u'][num_train + num_test:num_train + num_test + time_lag]
    dof_v = f['dof_v'][num_train + num_test:num_train + num_test + time_lag]

input = normalize_data(np.concatenate((dof_u , dof_v), axis=1), mean, std)
print(f'Input shape: {input.shape}')
print('Latent variables loaded and normalized.')

input_dim = input.shape[1]
# Predict to num_predictions
num_predictions = num_train - time_lag 
print(f'Predicting {num_predictions} time steps...')
predictions = predict(model, input, time_lag, num_predictions, device)
print('Predictions completed.')

# denormalize predictions
predictions = denormalize_data(predictions, mean, std)
# Save predictions
pred_path = os.path.join(predictions_dir, f'predictions_unseen_num{num_predictions}.h5')
with h5py.File(pred_path, 'w') as hf:
    hf.create_dataset('predictions', data=predictions)

print(f'Unseen predictions saved to {pred_path}.')