
# BIFOR

Repository for the paper "**A Background-Invariant Method for Generalizable Dog Re-identification in Non-Controlled Environments**".

## Table of Contents 

- [Overview](#overview) 
- [Requirements](#requirements) 
- [Datasets](#datasets) 
- [Weights](#weights) 
- [Training](#training) 
- [Evaluation](#evaluation) 

---
## Overview
This repository provides everything needed to reproduce our experimental results. It includes two publicly available datasets:

**YT-BB-Dog:** A short-term dataset with 2,723 dogs and 27,036 images. 

**Sibetan:** A long-term dataset featuring 59 dogs and 1,755 images.

We provide here the code to train and evaluate BIFOR (Background Invariant Feature Extractor), our approach for Generalizable long-term Re-Identification of dogs in uncontrolled environments. Training on the YT-BB-Dog and testing on Sibetan, BIFOR achieves SOTA results, with **82.7%** of *Rank-1* and **69.8%** of *mAP*.

---

## Requirements
### Libraries
We trained our models on python 3.8.10 using torch 2.3.1 and cuda 12.1. We recommend using either Python's `venv` or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create your environment.

1. **Set up your environment** 
Create your Python environment using `venv` or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html). 

2. **Install PyTorch** 
Install PyTorch with CUDA 12.1: `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121` 

Alternatively, search for your preferred library version in the [Pytorch website](https://pytorch.org/get-started/previous-versions/).

 3. **Install Remaining Dependencies**  
Download the remaining libraries using the command `pip install -r requirements.txt`.
 
 ## Datasets
 To download the YT-BB-Dog and the Sibetan, please refer to the link provided in our paper in the section **Declarations > Data Availability**. The datasets are in the folder "Data_BIFOR/Datasets").

 Please provide the path to the dataset in the parameter `data_dir` of your configuration file (refer to `config.yaml` as an example). The parameter `dataset_name` should be set as `"YT-BB-Dog"` or `"Sibetan"`.

## Weights
Our trained model BIFOR is provided in the zip file "f(2).zip" located in the folder "Data_BIFOR/Weights/". To load the model, set the weights path in the parameter `ckpt_path` and set the parameter `load_model` to `True`.

If you want to retrain, please download the model "Data_BIFOR/Weights/f(1).zip" and set its path in the parameter `sampler_path`.

## Training

To train the model, run the command:

    python main.py --config=[CONFIG PATH]
Make sure to adjust the training parameters in the configuration file as needed.

## Evaluation
To skip training and directly evaluate a model, set the `test_only` parameter of the configuration file to `True`. The same command used for training can be used for evaluation.

To evaluate on the Sibetan dataset, please set the parameter `gt_labels_path`, that corresponds to the path of the json file containing the ground truth labels for the dogs sequences. This file is provided inside the Sibetan dataset folder, named "gt_sibetan_no_mono_cluster.json".
