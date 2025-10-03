# Constructing Post-hoc Interpretations for Audio Classification Models

This repository contains code accompanying the paper *Constructing Post-hoc Interpretations for Audio Classification Models* (in preparation).

## 1. Overview

This project is focused on the task of constructing interpretations for machine learning models that classify audio data presented in the form of spectrograms. In this work, we explore the approach that involves masking spectrograms based on feature attribution maps and subsequently reconstructing the signal. The methods Saliency, Grad-CAM and SHAP are employed for feature attribution maps generation. The effectiveness of this pipeline is evaluated in terms of the fidelity of the intepretations to the model's behavior and their perceptual simplicity. Experiments are conducted with four different types of masks and with the addition of three kinds of background noise.

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
├── samples/                          # Audio samples for experimentation
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
├── illustration.ipynb                # Generating interpretation illustrations and audios
├── supplementary/                    # Supplementary materials for the paper
├── requirements.txt
└── README.md
```

## 3. Correspondence between audio files in this repository and ESC-50 dataset

The `samples` directory contains 5 audio files sourced from the ESC-50 dataset [9]. The following table maps the filenames used in this repository to their corresponding filenames in the original dataset.

| File in the `samples` directory   | File in the ESC-50 dataset   |
| --------------------------------- | ---------------------------- |
|  car_horn.wav                     |  2-100648-A-43.wav           |
|  clapping.wav                     |  2-76408-D-22.wav            |
|  crow.wav                         |  1-56234-A-9.wav             |
|  dog.wav                          |  3-157695-A-0.wav            |
|  sea_waves.wav                    |  4-182613-A-11.wav           |
|  cat.wav                          |  5-177614-A-5.wav            |


## 4. References
- [1]: Paissan F., Ravanelli M., Subakan C. [Listenable maps for audio classifiers]((https://doi.org/10.48550/arXiv.2403.13086)). arXiv preprint arXiv:2403.13086 (2024).
- [2]: [LMAC code](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50/interpret).
- [3]: [captum repository](https://github.com/meta-pytorch/captum).
- [4]: [pytorch-grad-cam repository](https://github.com/jacobgil/pytorch-grad-cam)
- [5]: [lime repository](https://github.com/marcotcr/lime)
- [6]: [SHAP repository](https://github.com/shap/shap)
- [7]: Hedström, Anna, et al. [Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond](https://www.jmlr.org/papers/v24/22-0142.html). Journal of Machine Learning Research 24.34 (2023): 1-11.
- [8]: Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. [Panns: Large-scale pretrained audio neural networks for audio pattern recognition](https://doi.org/10.1109/TASLP.2020.3030497). IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.
- [9]: K. J. Piczak. [ESC: Dataset for Environmental Sound Classification](https://dx.doi.org/10.1145/2733373.2806390). Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia (2015).
- [10]: [White noise by theundecided (freesound.org)](https://freesound.org/people/theundecided/sounds/165058/)
- [11]: [Room Tone Office Industrial Ambience 01 by mzui (freesound.org)](https://freesound.org/people/mzui/sounds/203297/)
- [12]: [Horse_Whinny.wav by foxen10 (freesound.org)](https://freesound.org/people/foxen10/sounds/149024/)
