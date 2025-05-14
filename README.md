# OSC_LDC_ML Project

## Directory Structure

Upon running a script, the entire directory structure will be initialized

```
OSC_LDC_ML/
├── anim/
├── checkpoints/
├── configs/            # Directory containing config jsons
│   ├── data.json
│   ├── misc.json
│   ├── train.json
│   ├── transformer.json
├── data/               # Directory for input .h5 data files
├── figs/               # Directory for saving plots and figures
├── latent/             # Directory for latent space files and configurations
├── lib/                # Directory containing function scripts
│   ├── dls.py
│   ├── eval.py
│   ├── transformer.py
├── models/
├── ldc_train.py        # Training script
├── ldc_analysis.ipynb  # Analysis notebook
├── README.md
└── requirements.txt
```

## Required Libraries:
Install the dependencies using:
```bash
pip install -r requirements.txt
```