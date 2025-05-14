#!/bin/bash 
#SBATCH --account PAA0008
#SBATCH --job-name LDC_model_train
#SBATCH --nodes=1 
#SBATCH --time=02:00:00 
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1 
#SBATCH --mail-type=ALL
#SBATCH --output=myjob.out

module load miniconda3/24.1.2-py310

cd $SLURM_SUBMIT_DIR
source activate /users/PAA0008/morlebcaton311/torch_env

cp ldc_train.py $TMPDIR
cp -r configs $TMPDIR
cp -r lib $TMPDIR
cp dls_funcs.py $TMPDIR
cp -r /fs/scratch/PAA0008/OSC_LDC_ML/latent $TMPDIR

cd $TMPDIR
python -u ldc_train.py

# Copy the results back to the original directory
cd $SLURM_SUBMIT_DIR
cd ..
mkdir $SLURM_JOB_ID
cp -R $TMPDIR/models $SLURM_JOB_ID