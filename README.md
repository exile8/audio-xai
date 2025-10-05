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

## 3. Methodology

The methodology in this project follows a three-stage workflow:

1. **Attribution**

Feature attribution maps are generated for each input sample using popular feature attribution methods:
- ***Saliency***,
- ***Grad-CAM***, 
- ***LIME***,
- ***SHAP***.

These maps highlight the time-frequency components of spectrograms which influence the model's decision the most.

2. **Masking**

The obtained attribution maps are then used to create *binary or continuous masks*. Each mask preserves or removes regions of spectrogram according to importance values. The masking strategies include:
- ***binarization***,
- ***minmax normalization***,
- ***sigmoid***,
- ***topK%***.

The masked spectrograms are ***reconstructed back into audio signals***, allowing for perceptual evaluation.

3. **Evaluation**

The fidelity and perceptual simplicity of each interpretation are assessed using the set of metrics.

This approach is strongly inspired by the work of Paissan, et al [1]. However, this project focuses on using ***standard inerpretation methods*** instead of an encoder-based setup and explores ***multiple masl types***.

## 4. Environment

* OS: Ubuntu 22.04
* NVIDIA RTX A6000 GPU
* Python 3.10.12
* PyTorch 2.6.0, CUDA 11.8

To install dependencies:
```bash
pip install -r requirements.txt
```

## 5. `audiointerp` module

## 6. Example usage

```python
import torch, torchaudio
from audiointerp.model.cnn14 import TransferCnn14
from audiointerp.interpretation.gradcam import GradCAMInterpreter
from audiointerp.processing.spectrogram import LogMelSTFTSpectrogram
from audiointerp.predict import Predict

# load audio
wav, _ = torchaudio.load("samples/crow.wav")

# Normalize
abs_max = wav.abs().max()
if abs_max != 0:
    wav /= abs_max

# feature extractor (Mel, dB)
sr = 32000
n_fft = 1024
hop_length = 320
win_length = 1024
n_mels = 64
f_min = 50
f_max = 14000
top_db = None

feature_extractor = LogMelSTFTSpectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    sample_rate=sr, n_mels=n_mels, f_min=f_min, f_max=f_max, top_db=top_db,
    return_phase=True, return_full_db=True
)

# pretrained model
model = TransferCnn14(50, 64)
model.load_state_dict(torch.load("weights/logmel_cnn14.pth"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Grad-CAM interpreter
predict_gradcam_mel = Predict(model, feature_extractor, interp_method_cls=GradCAMInterpreter, interp_method_kwargs={"target_layers": [model.base.conv_block6.conv2]}, device=device)

# interpret the sample
predict_gradcam_mel.predict(
    wav=wav, wav_name="crow_gradcam", sr=sr, feature_type="mel",
    silence_val=-100, fmin=50, fmax=14000,
    save_root="sample_pred", model_type="mel"
)
```

## 7. Reproducing the results
1. **Prepare the dataset**
   
The experiments were conducted on the *ESC-50* [9] dataset.
- Download the dataset [here](https://github.com/karolpiczak/ESC-50);
- Unpack and place the dataset in the desired directory.
In the notebooks, you will find fields where you can specify the path to this directory.

2. **Download pretrained weights (optional)**
   
The experiments use a pretrained model *Cnn14* [8]. The pipeline supports automatic downloading of pretrained weights. Alternatively, the weights can be downloaded manually:
```bash
wget https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth -P pretrained
```
The code will detect the weights in the `pretrained` directory automatically.

3. **Train models**
   
To train the models, use the notebooks `train_model_stft.ipynb` and `train_model_mel.ipynb`. The provided notebooks fine-tune pretrained model on STFT and Mel spectrogram respresentations, respectively.
> All hyperparameters and experimental settings are explicitely defined within each notebook.

During training, the *SpecAugment* [9] method is used.

The resulting models are automatically saved to `weights`.

4. **Run the experiments**

- ***Clean data:***

Experiments with uncontaminated data are performed for both STFT and Mel models. Use the notebooks `experiments_stft.ipynb` and `experiments_mel.ipynb`.

- ***Background noise:***

Experiments with background noise are conducted for the Mel model.

The `noises` directory contains 5-second fragments representing different types of background noise, including *synthetic white noise*, *industrial room ambience* and *horse sounds (hoof beats and heighing)*. Each fragment was extracted from publically available audio recordings on Freesound [11, 12, 13].

To run the experiments, use the notebooks `experiments_mel_noise1.ipynb`, `experiments_mel_noise2.ipynb` and `experiments_mel_noise3.ipynb`.

3. **Explore the results**

The results obtained from the previous step are stored in the `results` directory, following the structure:
```
results/
└── <model_name>/
    └── <inetrpretation_method_name>_<experiment_name>/
        ├── attributions/
        └── csvs/
```
Each experiment directory contains the subfolders:
- `attributions` - feature attribution maps generated by the corresponding interpretation method.
- `csvs` - metric values for each mask type, stored as csv-files.

You can analyze and visualize the results using the following notebooks:
- `tables.ipynb` - visualize the results as csv-based tables. You can explore:
    - results for each mask within a given interpretation method,
    - results for each interpretation method given a mask type,
    - performance on correctly and incorrectly classified examples,
    - summary tables consistent with the ones provided in the paper.
- `illustration.ipynb` - generate visual and audible interpretations for an audio sample, e.g. from the `samples` directory. This notebooks also reproduces the illustrative figure from the paper.

## 8. Supplementary materials

The `supplementary` directory contains additional materials referenced in the paper:
- a PDF version of the complete set of tables generated using `tables.ipynb`,
- the high-resolution illustrartion from the paper,
- the full set of visual and audible interpretations for the sample `samples/cat.wav`.

## 9. Extending the project
  
## 10. Correspondence between audio files in this repository and ESC-50 dataset

The `samples` directory contains 5 audio files sourced from the ESC-50 dataset [9]. The following table maps the filenames used in this repository to their corresponding filenames in the original dataset.

| File in the `samples` directory   | File in the ESC-50 dataset   |
| --------------------------------- | ---------------------------- |
|  car_horn.wav                     |  2-100648-A-43.wav           |
|  clapping.wav                     |  2-76408-D-22.wav            |
|  crow.wav                         |  1-56234-A-9.wav             |
|  dog.wav                          |  3-157695-A-0.wav            |
|  sea_waves.wav                    |  4-182613-A-11.wav           |
|  cat.wav                          |  5-177614-A-5.wav            |


## 11. References
[1]: Paissan F., Ravanelli M., Subakan C. [Listenable maps for audio classifiers]((https://doi.org/10.48550/arXiv.2403.13086)). arXiv preprint arXiv:2403.13086 (2024).

[2]: [LMAC code](https://github.com/speechbrain/speechbrain/tree/develop/recipes/ESC50/interpret).

[3]: [captum repository](https://github.com/meta-pytorch/captum).

[4]: [pytorch-grad-cam repository](https://github.com/jacobgil/pytorch-grad-cam).

[5]: [lime repository](https://github.com/marcotcr/lime).

[6]: [SHAP repository](https://github.com/shap/shap).

[7]: Hedström, Anna, et al. [Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond](https://www.jmlr.org/papers/v24/22-0142.html). Journal of Machine Learning Research 24.34 (2023): 1-11.

[8]: Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. [Panns: Large-scale pretrained audio neural networks for audio pattern recognition](https://doi.org/10.1109/TASLP.2020.3030497). IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

[9]: Park, Daniel S., et al. [Specaugment: A simple data augmentation method for automatic speech recognition](https://doi.org/10.21437/Interspeech.2019-2680). Interspeech 2019 (2019): 2613.

[10]: K. J. Piczak. [ESC: Dataset for Environmental Sound Classification](https://dx.doi.org/10.1145/2733373.2806390). Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia (2015).

[11]: [White noise by theundecided (freesound.org)](https://freesound.org/people/theundecided/sounds/165058/).

[12]: [Room Tone Office Industrial Ambience 01 by mzui (freesound.org)](https://freesound.org/people/mzui/sounds/203297/).

[13]: [Horse_Whinny.wav by foxen10 (freesound.org)](https://freesound.org/people/foxen10/sounds/149024/).
