# nohup /home/HDD/cmorton/OSC_LDC_ML/.ldc_venv/bin/python3 -u /home/HDD/cmorton/OSC_LDC_ML/ldc_train.py > train.log &
# nohup /home/cmorton/Desktop/beta-Variational-autoencoders-and-transformers-for-reduced-order-modelling-of-fluid-flows/.venv/bin/python3 -u /home/HDD/cmorton/OSC_LDC_ML/ldc_train.py > train.log &

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

# import matplotlib.pyplot as plt # matplotlib not found
import time
# from torchsummary import summary
from tqdm import tqdm
from datetime import datetime

from lib.eval import * # Imports analysis functions
from lib.transformer import TransformerEncoderModel, make_Sequence, train_model, normalize_data, denormalize_data

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
time_lag = config["transformer"]["time_lag"]
d_model = config["transformer"]["d_model"]
nhead = config["transformer"]["nhead"]
num_layers = config["transformer"]["num_layers"]
lr = config["train"]["lr"]
num_epochs = config["train"]["num_epochs"]
patience = config["train"]["patience"]
train_ahead = config["train"]["train_ahead"]
num_train = config["train"]["num_train"]
num_test = config["train"]["num_test"]
test_split = config["train"]["test_split"]
val_split = config["train"]["val_split"]
batch_size = config["train"]["batch_size"]
ram_available = config["misc"]["ram_available"]

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
latent_config_file = 'latent/' + latent_name + '_config.pkl'
latent_metrics = 'latent/' + latent_name + '_metrics.txt'

## ====================================== Load Data ==========================================

with open(latent_config_file, 'rb') as f:
    latent_config = pickle.load(f)

num_dofs = 2*latent_config.num_gfem_nodes * latent_config.dof_node
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
std_data = np.std(reshaped_data, axis=(0))
mean_data = np.mean(reshaped_data, axis=(0))

normalized_data = normalize_data(reshaped_data, mean_data, std_data)
print(normalized_data.shape)

print(np.max(normalized_data), np.min(normalized_data))

# Save normalization parameters
np.savez(normalization_params_save_path, mean=mean_data, std=std_data)
print(f"Normalization parameters saved to {normalization_params_save_path}")

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

del reshaped_data, dof_u, dof_v
del train_data, test_data

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

## ====================================== Model Train ==========================================
# Initialize the model and move it to the GPU
input_dim = X_train.shape[-1]
del X_train, Y_train, X_test, Y_test

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

# Save training and test losses to file
with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# # Plot the training and test losses
# plt.figure(figsize=(4, 2.5))
# plt.plot(results["train_losses"], label='Training Loss')
# plt.plot(results["test_losses"], label='Test Loss')
# plt.title('Training Curve')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.grid()
# plt.yscale('log')
# plt.show()
# print number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params}")