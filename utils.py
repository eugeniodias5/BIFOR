import os, json

import torch
import matplotlib.pyplot as plt
import pandas as pd

from collections import OrderedDict


def compute_mean_features(features, labels):
    '''
    Compute the mean feature vector for each class.

    Args:
        labels (Tensor): A tensor of shape (N,) containing the labels for each vector.
        features (Tensor): A tensor of shape (N, D) containing the features.

    Returns:
        Tensor: A tensor of shape (C, D) containing the mean feature vector for each class.
    '''

    unique_labels = sorted(set(labels.tolist()))

    mean_features = []
    
    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        mean_vector = features[label_indices].mean(axis=0)
        
        mean_features.append(mean_vector)

    mean_features = torch.stack(mean_features)
    
    return mean_features

def _compute_cosine_distance_matrix(features):
    '''
    Compute the cosine distance matrix.

    Args:
        features (Tensor): A tensor of shape (N, N) containing the features.

    Returns:
        Tensor: A tensor of shape (N, N) containing the cosine distances between vectors.
    '''
    # Normalize the feature vectors
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    # Compute the cosine similarity matrix
    cosine_sim = torch.mm(features_norm, features_norm.t())
    # Compute the cosine distance
    cosine_dist = 1 - cosine_sim

    return cosine_dist

def evaluate_cmc_map(features, labels, max_k):
    """
    Evaluate CMC and mAP using the cosine distance matrix.

    Args:
        features (Tensor): A tensor of shape (N, N) containing the features.
        labels (Tensor): A tensor of shape (N,) containing the labels for each vector.
        max_k (int): The maximum rank for calculating CMC.

    Returns:
        float: The mean Average Precision (mAP).
        Tensor: The CMC curve up to rank max_k.
    """
    
    # Calculating the cosine distances 
    cos_dist = _compute_cosine_distance_matrix(features) 
    N = cos_dist.size(0)

    # Filling diagonal to avoid self-matches
    cos_dist.fill_diagonal_(torch.inf)

    if N < max_k:
        max_k = N
        print(f"The number of test samples is small, changing the test top-k to {N}")

    # Sorting the cosine distances matrix
    indices = torch.argsort(cos_dist, axis=1)
    # Drop the self-matches
    indices = indices[:, :-1]

    matches = (labels[indices] == labels.unsqueeze(1)).float()

    # Computing cmc and mAP
    all_cmc = []
    all_ap = []
    num_test = 0.

    for idx in range(N):
        raw_cmc = matches[idx]
        if not torch.any(raw_cmc):
            continue
        
        # Computing cmc
        cmc = raw_cmc.cumsum(dim=0)
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_k])
        num_test += 1

        # Computing AP
        num_rel = matches[idx].sum()
        tmp_cmc = matches[idx].cumsum(dim=0)
        tmp_cmc = tmp_cmc / torch.arange(1, len(tmp_cmc) + 1)
        tmp_cmc = tmp_cmc * matches[idx]
        AP = tmp_cmc.sum() / num_rel if num_rel else 0
        all_ap.append(AP)

    assert num_test > 0, "No images were matched"

    cmc_scores = torch.stack(all_cmc)
    cmc_scores = cmc_scores.sum(0) / num_test
    all_ap = torch.tensor(all_ap)
    map = all_ap.mean()

    return map, cmc_scores

def save_closest_imgs_json(img_paths, features, save_path, max_closest=50):
    """
    Save a JSON file with each image path as a key and a list of paths to its closest images as the value,
    limited to a maximum number of closest images as specified by max_closest.

    Args:
        img_paths (list of str): Paths to the images.
        features (Tensor): A tensor of shape (N, N) containing the features.
        save_path (str): The path to save the JSON file.
        max_closest (int): Maximum number of closest images to save for each image.
    """

    closest_images = {}
    
    cosine_dist = _compute_cosine_distance_matrix(features)

    # Filling diagonal to avoid self-matches
    cosine_dist.fill_diagonal_(2)
    
    for i, path in enumerate(img_paths):
        # Get the indices of the sorted distances, then select the top closest (excluding itself)
        _, indices = torch.sort(cosine_dist[i])
        
        # Convert tensor indices to list of paths. Limit the number of closest images based on max_closest
        closest_paths = [img_paths[idx] for idx in indices.cpu().numpy()[:max_closest]]
        
        # Assign the sorted and limited list to the corresponding image path in the dictionary
        closest_images[path] = closest_paths

    # Save the dictionary as a JSON file
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(closest_images, f, indent=4)

    print(f"Saved closest images JSON to {save_path}")

def save_metrics(save_path, y_loss=None, y_acc=None, y_margin=None, y_map=None, y_cmc=None):
    '''
    Save the metrics data to a csv file and the performance graphs to png files.

    Args:
        save_path (str): The path to save the metrics data and performance graphs.
        y_loss (dict): Dictionary containing the loss values for train and test sets.
        y_acc (dict): Dictionary containing the accuracy values for train and test sets.
        y_margin (dict): Dictionary containing the margin values for train and test sets.
        y_map (dict): Dictionary containing the mAP values for test set.
        y_cmc (dict): Dictionary containing the CMC values for test set.

    '''


    # Drawing performances graphs
    if y_loss is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(y_loss['train'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss.png'))
        plt.close()

    if y_acc is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(y_acc['train'], label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'accuracy.png'))
        plt.close()

    if y_margin is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(y_margin['train'], label='Train Margin')
        plt.xlabel('Epoch')
        plt.ylabel('Margin')
        plt.title('Margin Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'margin.png'))
        plt.close()

    if y_map is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(y_map['test'], label='mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('mAP Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'test_map.png'))
        plt.close()

    if y_cmc is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(y_cmc['test']) + 1), y_cmc['test'], label='CMC Curve')
        plt.xlabel('Rank (k)')
        plt.ylabel('Matching Accuracy')
        plt.title('CMC Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'cmc_curve.png'))
        plt.close() 

    # Saving performances data to a csv file
    epochs = list(range(1, len(y_loss['train']) + 1))
    data = {
        'Epoch': epochs,
        'Train Loss': y_loss['train'],
        'Train Accuracy': y_acc['train'],
        'Train Margin': y_acc['train'],
        'Test mAP': y_map['test'],
    }

    # Saving the dataframe as a csv file
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, 'metrics_data.csv'), index=False)

    # Saving the cmc ranks
    cmc_data = {}
    for k, v in y_cmc.items():
        cmc_data[k] = v

    df = pd.DataFrame(cmc_data)
    df.to_csv(os.path.join(save_path, 'cmc_data.csv'), index=False)

def save_network(network, epoch_label, save_path, device=0, optimizer=None, scheduler=None):
    filename = f"{epoch_label}.pth"
    optimizer_filename = f"{epoch_label}_optimizer.pth"
    scheduler_filename = f"{epoch_label}_scheduler.pth"

    # Clean up previous checkpoints if necessary
    if "last_epoch" in epoch_label:
        # Erasing the ckpt of the last epoch in the results folder
        for file in os.listdir(save_path):
            if file.endswith(".pth") and "last_epoch" in file:
                os.remove(os.path.join(save_path, file))

    elif "best_epoch" in epoch_label:
        # Erasing the ckpt of the best epoch in the results folder
        for file in os.listdir(save_path):
            if file.endswith(".pth") and "best_epoch" in file:
                os.remove(os.path.join(save_path, file))

    # Saving the network state
    save_model_path = os.path.join(save_path, filename)
    network.to("cpu")  # Move network to CPU before saving
    torch.save(network.state_dict(), save_model_path)

    # Optionally save the optimizer state
    if optimizer is not None:
        save_optimizer_path = os.path.join(save_path, optimizer_filename)
        torch.save(optimizer.state_dict(), save_optimizer_path)

    # Optionally save the scheduler state
    if scheduler is not None:
        save_scheduler_path = os.path.join(save_path, scheduler_filename)
        torch.save(scheduler.state_dict(), save_scheduler_path)

    # Move network back to the original device
    network.to(device)

    print(f"Saved model to {save_model_path}")
    if optimizer is not None:
        print(f"Saved optimizer state to {save_optimizer_path}")
    if scheduler is not None:
        print(f"Saved scheduler state to {save_scheduler_path}")

def load_model(model, ckpt_path, model_name=None, device='cpu'):
    """
    Load a model checkpoint directly from a given path.

    Parameters:
    - model: The model object to which the checkpoint will be loaded.
    - model_path: The full path to the checkpoint file.
    - device: The device to load the model onto ('cpu' or 'cuda').

    Returns:
    - The model loaded with the checkpoint.
    """

    # Check if the checkpoint file exists
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'")

    # Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    if 'state_dict' in ckpt: state_dict = ckpt['state_dict']
    else: state_dict = ckpt
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Ensure the model is on the correct device
    model.to(device)

    return model

def resume_training(model, results_folder, device='cpu', optimizer=None, scheduler=None, epoch_label='last'):
    """
    Resume training by loading a model and optionally its optimizer and scheduler states from checkpoints.

    Parameters:
    - model: The model object to which the checkpoint will be loaded.
    - results_folder: The directory path where checkpoints are stored.
    - device: The device to load the model.
    - optimizer: The optimizer.
    - scheduler: The learning rate scheduler.
    - epoch_label: The label to identify which checkpoint to load ('last_epoch', 'best_epoch', etc.).

    Returns:
    - The model loaded with the checkpoint.
    - The optimizer loaded with the checkpoint, if provided.
    - The scheduler loaded with the checkpoint, if provided.
    """

    model_ckpt_path = os.path.join(results_folder, f"{epoch_label}.pth")
    optimizer_ckpt_path = os.path.join(results_folder, f"{epoch_label}_optimizer.pth")
    scheduler_ckpt_path = os.path.join(results_folder, f"{epoch_label}_scheduler.pth")

    # Load model checkpoint
    if not os.path.isfile(model_ckpt_path):
        raise FileNotFoundError(f"No model checkpoint found at '{model_ckpt_path}'")
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.to(device)

    # Load optimizer checkpoint if provided
    if optimizer is not None and os.path.isfile(optimizer_ckpt_path):
        optimizer.load_state_dict(torch.load(optimizer_ckpt_path, map_location=device))

    # Load scheduler checkpoint if provided
    if scheduler is not None and os.path.isfile(scheduler_ckpt_path):
        scheduler.load_state_dict(torch.load(scheduler_ckpt_path, map_location=device))

    return model, optimizer, scheduler
