# OSC_LDC_ML Project

## Directory Structure

```
OSC_LDC_ML/
├── data/               # Directory for input data files (e.g., HDF5 files)
├── latent/             # Directory for latent space files and configurations
├── figs/               # Directory for saving plots and figures
├── anim/               # Directory for saving animations
├── checkpoints/        # Directory for saving model checkpoints
├── models/             # Directory for saving trained models
├── ldc_train.py        # Training script
├── ldc_analysis.ipynb  # Analysis notebook
├── analysis_funcs.py   # Misc. functions for calculating TKE, etc.
├── dls_funcs.py        # Functions for computing DLS latent
└── README.md           # Project documentation
```

### Required Libraries:
Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Workflow

1. **Data Preparation**:
   - Place the required `.h5` data files in the `data/` directory.
   - Ensure the latent space files are generated or compute them using the notebook.

2. **Model Training**:
   - Run `ldc_train.py` to train the Transformer Encoder model.
   - Monitor the training loss and test loss during execution.

3. **Analysis**:
   - Use `ldc_analysis.ipynb` to evaluate the model's performance.
   - Visualize predictions, attention weights, and error metrics.

4. **Visualization**:
   - Generate plots for TKE, RMS, and PSD.
   - Create animations for predicted and true flow fields.
