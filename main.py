import os

import argparse
import yaml
import random

import numpy as np

import torch
from torch import optim
from torch import nn

from torchvision import transforms

from data.offline_sampler import BackgroundSampler, OfflineSampler
from data.online_sampler import OnlineBackgroundSampler, RandomIdentitySampler
from data.yt_dataset import YtDogDataset
from data.sibetan_dataset import SibetanDataset

from models.Bifor import Bifor
from models.BackgroundNet import BackgroundNet
from train import train_model
from test import test

from utils import load_model, resume_training

version = torch.__version__


# Adding random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

random.seed(seed)
np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)


def main() -> None:
    ######################################################################
    # Loading config file

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="path to config file"
    )
    args = parser.parse_args()

    # Read config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Check if gpu ids are available
    use_gpu = torch.cuda.is_available()

    device = "cpu"
    gpu_ids = config["gpu_ids"]
    if use_gpu:
        # Convert gpu_ids to list
        gpu_ids = [int(gid) for gid in gpu_ids.split(",") if int(gid) >= 0]

        if torch.cuda.is_available(): device = f"cuda:{gpu_ids[0]}"


    ######################################################################
    # Load Data

    transform_train_list = [
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    transform_test_list = [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    data_transforms = {
        "train": transforms.Compose(transform_train_list),
        "test": transforms.Compose(transform_test_list),
    }

    image_datasets = {}
    if config["dataset_name"] == "YT-BB-Dog":
        image_datasets["train"] = YtDogDataset(
            os.path.join(config["data_dir"], "train"), data_transforms["train"]
        )
        image_datasets["test"] = YtDogDataset(
            os.path.join(config["data_dir"], "test"), data_transforms["test"]
        )

    elif config["dataset_name"] == "Sibetan":
        image_datasets["test"] = SibetanDataset(config["data_dir"], gt_path=config["gt_labels_path"], transform=data_transforms["test"])
        config["test_only"] = True
        print("Training code is not implemented for Sibetan dataset. Procceding to test.")
        print("Please implement the training code if you want to fine tune on the Sibetan dataset.")

    else:
        raise ValueError(f"Invalid dataset name {config['dataset_name']}. Pleease choose between 'YT-BB-Dog' or 'Sibetan'.")
    
    test_dataloader = torch.utils.data.DataLoader(
            image_datasets["test"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )        

    ######################################################################
    # Training the model

    # Loading the model
    model_name = config['model_name']
    
    model = Bifor()

    if config['load_model']:
        model = load_model(model=model, ckpt_path=config['ckpt_path'], model_name=model_name, device=device)

    if use_gpu and len(gpu_ids) > 1:
        # Parallelize the model
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)

    if config['test_only']:
        print("TESTING THE MODEL")
        test(model=model, dataloader=test_dataloader, device=device, cfg=config)

    else:
        if config["sampler"] == "bifor":
            '''
            We load the model that re-identifies dogs based on the background similarity
            '''
            sampler_net = BackgroundNet.load_from_checkpoint(config["sampler_path"], map_location=device)
            sampler_net.eval()

            sampler = None

            if config["online"]:
                sampler = OnlineBackgroundSampler(
                    dataset=image_datasets["train"], 
                    batch_size=config["batch_size"],
                    num_instances=config["num_instances"],
                    model=sampler_net,
                    device=device,
                    num_workers=config["num_workers"]
                )
            else:
                sampler = BackgroundSampler(
                    dataset=image_datasets["train"],
                    batch_size=config["batch_size"],
                    model=sampler_net,
                    device=device,
                    num_workers=config["num_workers"],
                )
        elif config["sampler"] == "random":
            if config["online"]:
                sampler = RandomIdentitySampler(
                    data_source=image_datasets["train"],
                    batch_size=config["batch_size"],
                    num_instances=config["num_instances"],
                )
            else:
                sampler = OfflineSampler(
                    dataset=image_datasets["train"],
                    batch_size=config["batch_size"],
                )

        else:
            raise ValueError(f"Invalid sampler value {config['sampler']}. Expected 'random' or 'bifor.")

        train_dataloader = torch.utils.data.DataLoader(
                image_datasets["train"],
                batch_size=config["batch_size"],
                sampler=sampler,
                shuffle=False,
                num_workers=config["num_workers"],
            )
        
        train_steps = len(train_dataloader) * config["num_epochs"]

        # Defining a adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        # Defining a scheduler with linear warmup of 10% of total epochs and then cosine decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            total_steps=train_steps,
            pct_start=config["warmup"],  # 10% of training for the warm-up phase
            anneal_strategy="cos",  # Cosine annealing
        )

        if config["resume"]:
            model, optimizer, scheduler = resume_training(
                model, 
                results_folder=config["load_resume_dir"], 
                device=device,
                optimizer=optimizer, 
                scheduler=scheduler,
                epoch_label='last'
                )

        # Training the model
        model = train_model(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            test_dataloader,
            device,
            config
        )

        # We save the configurations to the results file
        cfg_save_path = os.path.join(config["save_dir"], "config.yaml")

        with open(cfg_save_path, "w") as cfg_file:
            yaml.dump(config, cfg_file)


if __name__ == "__main__":
    main()
    