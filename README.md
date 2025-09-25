# Constructing Post-hoc Interpretations for Audio Classification Models

This repository contains code accompanying the paper *Constructing Post-hoc Interpretations for Audio Classification Models* (in preparation).

## 1. Overview

This project is focused on the task of constructing interpretations for machine learning models that classify audio data presented in the form of spectrograms. In this work, we explore the approach that involves masking spectrograms based on feature attribution maps and subsequently reconstructing the signal. The methods Saliency, Grad-CAM and SHAP are employed for feature attribution maps generation. The effectiveness of this pipeline is evaluated in terms of the fidelity of the intepretations to the model's behavior and their perceptual simplicity. Experiments are conducted with different types of masks and with the addition of various kinds of background noise.

## 2. Project structure
```text
├── audiointerp/                      # Main module with core functionality
│   ├── dataset/                      # Dataset handling
│   ├── processing/                   # Feature extraction and inversion
│   ├── model/                        # Model handling
│   ├── interpretation/               # Interpretation methods
│   ├── fit.py                        # Model training routines
│   ├── metrics.py                    # Metrics for evaluating interpretations' quality
|   └── predict.py                    # Inference, interpretation + metric logging
├── noises/                           # Background noise samples
├── pretrained/                       # Pretrained backbones for initialization
|── weights/                          # Final trained models for experiments
├── train_model_stft.ipynb            # Training model on STFT spectrograms
├── train_model_mel.ipynb             # Training model on mel-spectrograms
├── experiments_stft.ipynb            # Running experiments with the STFT model
├── experiments_mel.ipynb             # Running experiments with the mel model
├── experiments_mel_noise1.ipynb      # Mel model (horse sounds)
├── experiments_mel_noise2.ipynb      # Mel model (room ambience)
├── experiments_mel_noise3.ipynb      # Mel model (white noise)
├── results/                          # Interpetation results and metric logs
├── tables.ipynb                      # Tables summarizing interpetation results and evaluations
├── supplementary/                    # Supplementary materials for paper
├── requirements.txt
└── README.md
```