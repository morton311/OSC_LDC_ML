#!/bin/bash 
#SBATCH --account PAA0008
#SBATCH --job-name Python_ExampleJob 
#SBATCH --nodes=1 
#SBATCH --time=00:10:00 
#SBATCH --gpus-per-node=1 

source activate /users/PAA0008/morlebcaton311/torch_env

python << EOF
import torch
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
EOF