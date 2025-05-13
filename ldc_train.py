# nohup python -u /home/HDD/cmorton/Rohit_LDC/ldc_data_for_Caleb/ldc_train.py > out.log &

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

import matplotlib.pyplot as plt
import time
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime

from OSC_LDC_ML.lib.dls import gfem_2d, gfem_recon  # Explicitly import required functions
from OSC_LDC_ML.lib.eval import * # Imports analysis functions
from lib.transformer import TransformerEncoderModel, make_Sequence, train_model

print('Libraries loaded.')

# load configs
config_path = 'configs/'
with open(config_path + 'data.json', "r") as f:
    data_cfg = json.load(f)
with open(config_path + 'transformer.json', "r") as f:
    transformer_cfg = json.load(f)
with open(config_path + 'train.json', "r") as f:
    train_cfg = json.load(f)
with open(config_path + 'misc.json', "r") as f:
    misc_cfg = json.load(f)

# Access parameters from the JSON file
data_name = data_cfg["data_name"]
patch_size = data_cfg["patch_size"]
num_modes = data_cfg["num_modes"]
time_lag = transformer_cfg["time_lag"]
d_model = transformer_cfg["d_model"]
nhead = transformer_cfg["nhead"]
num_layers = transformer_cfg["num_layers"]
lr = train_cfg["lr"]
num_epochs = train_cfg["num_epochs"]
patience = train_cfg["patience"]
train_ahead = train_cfg["train_ahead"]
num_train = train_cfg["num_train"]
num_test = train_cfg["num_test"]
test_split = train_cfg["test_split"]
val_split = train_cfg["val_split"]
batch_size = train_cfg["batch_size"]
ram_available = misc_cfg["ram_available"]

print('num_train:', num_train)
print('num_test:', num_test)


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
latent_config = 'latent/' + latent_name + '_config.pkl'
latent_metrics = 'latent/' + latent_name + '_metrics.txt'

## ====================================== Load Data ==========================================

with open(latent_config, 'rb') as f:
    config = pickle.load(f)

num_dofs = 2*config.num_gfem_nodes * config.dof_node
max_coeffs = (num_train + num_test - time_lag - train_ahead, time_lag + train_ahead, num_dofs)
max_coeffs_prod = math.prod(max_coeffs)
print('Max coeffs:', max_coeffs_prod)
print('Bytes of Max coeffs:', max_coeffs_prod * 4)
print('GB of Max coeffs:', max_coeffs_prod * 4 / 1e9)


with h5py.File(latent_file, 'r') as f:
    print('Latent space shape:', f['dof_u'].shape)
    dof_u = f['dof_u'][:num_train + num_test]
    dof_v = f['dof_v'][:num_train + num_test]
    print('Train data shape:', dof_u.shape)

reshaped_data = np.concatenate((dof_u , dof_v), axis=1)
print(reshaped_data.shape)


## ====================================== Normalize Data ==========================================
# normalize data set with standard scaler
def normalize_data(data, mean, std):
    return (data - mean) / std
def denormalize_data(data, mean, std):
    return (data * std) + mean

std_data = np.std(reshaped_data, axis=(0))
mean_data = np.mean(reshaped_data, axis=(0))

normalized_data = normalize_data(reshaped_data, mean_data, std_data)
print(normalized_data.shape)

print(np.max(normalized_data), np.min(normalized_data))

# Save normalization parameters
np.savez(normalization_params_save_path, mean=mean_data, std=std_data)
print(f"Normalization parameters saved to {normalization_params_save_path}")
del reshaped_data, dof_u, dof_v

## ====================================== Organize Data ==========================================
# Split the data into training and validation sets
train_data = normalized_data[:num_train].astype(np.float32)
test_data = normalized_data[num_train:num_train + num_test].astype(np.float32)

print('Training set is shape:', train_data.shape)
print('Testing set is shape:', test_data.shape)

X_test, Y_test   = make_Sequence(time_lag,test_data)
X_train, Y_train = make_Sequence(time_lag,train_data)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

del train_data, test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU is available. Using CUDA.")
else:
    print("GPU is not available. Using CPU.")
print(f"Using device: {device}")

# Move data to the GPU
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

gc.collect()

X_train = torch.tensor(X_train, dtype=torch.float32)


# Data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

del X_train, Y_train, X_test, Y_test

## ====================================== Model Train ==========================================
# Initialize the model and move it to the GPU
input_dim = X_train.shape[-1]
model = TransformerEncoderModel(time_lag, input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Call the train_model function
results = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    patience=patience,
    device=device,
    model_dir=model_dir,
    data_name=data_name
)

# Plot the training and test losses
plt.figure(figsize=(4, 2.5))
plt.plot(results["train_losses"], label='Training Loss')
plt.plot(results["test_losses"], label='Test Loss')
plt.title('Training Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()
# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params}")