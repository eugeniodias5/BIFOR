import time, os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils import save_metrics, save_network
from test import test


# Training the model
y_loss = {"train": [], "test": []}
y_acc = {"train": []}
y_margin = {"train": []}
y_map = {"test": []}
y_cmc = {"test": []}


def train_model(model, optimizer, scheduler, train_dl, test_dl, device, cfg):  
    '''
    Args:
        model: Model to train.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        train_dl: Train dataloader.
        test_dl: Test dataloader.
        device: Device to run the model.
        cfg: Configuration dictionary (yaml file).
    
    '''

    dataset_size = len(train_dl.sampler)

    since = time.time()

    best_map = 0.0
    best_cmc = 0.0

    improvement_count = 0

    # Create the results save directory if it does not exist
    save_path = os.path.join(cfg["save_dir"], cfg["model_name"])
    os.makedirs(save_path, exist_ok=True)

    # Create logging file
    log_path = os.path.join(save_path, "train_log.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    for epoch in range(cfg["num_epochs"]):
        print(f"Epoch {epoch}/{cfg['num_epochs']}")
        logging.info(f"Epoch {epoch}/{cfg['num_epochs']}")
        print("-" * 10)
        logging.info("-"*10)
        
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0
        running_margin = 0.0

        # Iterate over data.
        for batch in tqdm(train_dl, desc="Processing training batch"):
            if cfg["online"]:
                anchors, labels = batch
            else:
                anchors, positives, negatives, labels = batch
                positives = positives.to(device)
                negatives = negatives.to(device)

            anchors = anchors.to(device)            
            labels = torch.tensor([int(l) for l in labels])
            labels = labels.to(device)

            batch_size = anchors.shape[0]

            if batch_size < cfg["batch_size"]:  # Next epoch
                continue

            optimizer.zero_grad()

            # Forward
            feats = model(anchors)

            if cfg["online"]:
                psim, nsim = hard_triplet_similarity(feats, labels)
            
            else:
                pos_feats = model(positives)
                neg_feats = model(negatives)

                psim = F.cosine_similarity(feats, pos_feats)
                nsim = F.cosine_similarity(feats, neg_feats)
            
            if psim is not None and nsim is not None:
                # Calculate the triplet loss
                triplet_loss = torch.sum(torch.nn.functional.relu(nsim + cfg["margin"] - psim))
                triplet_loss.backward()

                optimizer.step()
                scheduler.step()

                # Update running loss
                running_loss += triplet_loss.item()
                running_corrects += float(torch.sum(psim > nsim + cfg["margin"]))
                running_margin += float(torch.sum(psim - nsim))
        
        datasize = dataset_size // cfg["batch_size"] * cfg["batch_size"]
        epoch_loss = running_loss / datasize
        epoch_acc = running_corrects / datasize
        epoch_margin = running_margin / datasize

        print(
            "Train Loss: {:.4f}, Acc: {:.4f}, MeanMargin: {:.4f}, LR: {:.6f}".format(
                epoch_loss, epoch_acc, epoch_margin, scheduler.get_last_lr()[0]
            )
        )
        logging.info(
            "Train Loss: {:.4f}, Acc: {:.4f}, MeanMargin: {:.4f}, LR: {:.6f}".format(
                epoch_loss, epoch_acc, epoch_margin, scheduler.get_last_lr()[0]
            )
        )

        y_loss["train"].append(epoch_loss)
        y_acc["train"].append(epoch_acc)
        y_margin["train"].append(epoch_margin)
        
        if (epoch + 1) % cfg["test_every_n"] == 0:
            # Test the model
            test_res = test(model, test_dl, device, cfg)
            map = test_res["map"]
            cmc = test_res["cmc"]

            y_map["test"].append(map.item())

            logging.info(
                f"Test mAP: {map}, top-1: {cmc[0]}, top-5: {cmc[4]}, top-10: {cmc[9]}"
            )

            # Check by mAP if this is the best model
            if map > best_map:
                best_map = test_res["map"]
                best_cmc = test_res["cmc"]
                y_cmc["test"] = best_cmc

                print(
                    f"Best model found at epoch {epoch} with mAP: {best_map}, top-1: {cmc[0]}, top-5: {cmc[4]}, top-10: {cmc[9]}"
                )
                logging.info(
                    f"Best model found at epoch {epoch} with mAP: {best_map}, top-1: {cmc[0]}, top-5: {cmc[4]}, top-10: {cmc[9]}"
                )
                save_network(
                    network=model,
                    epoch_label=f"best_epoch_{epoch}",
                    save_path=save_path,
                    device=device,
                )
                improvement_count = 0

            else:
                improvement_count += cfg["test_every_n"]

        # Saving model's last weights
        save_network(
            network=model,
            epoch_label=f"last",
            save_path=save_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        epoch_time_elapsed = time.time() - since
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                epoch_time_elapsed // 60, epoch_time_elapsed % 60
            )
        )
        logging.info(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                epoch_time_elapsed // 60, epoch_time_elapsed % 60
            )
        )

        if improvement_count >= cfg["early_stop"]:
            print(f"Early stopping at epoch {epoch}")
            logging.info(f"Early stopping at epoch {epoch}")
            break


    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Saving the result graphs
    save_metrics(cfg["save_dir"], y_loss, y_acc, y_margin, y_map, y_cmc)

    return model


def hard_triplet_similarity(features, labels):
    '''
    Function to find the hard positives and negatives within a mini-batch.
    
    Args:
        features: Features extracted from the model.
        labels: Labels of the samples.

    Returns:
        psim: Similarity of the hard positives.
        nsim: Similarity of the hard negatives.

    '''

    # We find the hard positives and negatives
    norm_feats = F.normalize(features, p=2, dim=1)
    cos_similarity = torch.matmul(norm_feats, norm_feats.t())
    positive_mask = labels[:, None] == labels[None, :]
    negative_mask = labels[:, None] != labels[None, :]

    pos_similarities = cos_similarity * positive_mask.float()
    pos_similarities[~positive_mask] = 1 # Minimum value for cosine similarity
    hard_positives = torch.argmin(pos_similarities, axis=1)

    neg_similarities = cos_similarity.clone()
    neg_similarities[~negative_mask] = 0 # Maximum value for cosine similarity
    hard_negatives = torch.argmax(neg_similarities, axis=1)

    if hard_positives is not None and hard_negatives is not None:
        # We calculate the similarities
        psim = pos_similarities[torch.arange(features.shape[0]), hard_positives]
        nsim = neg_similarities[torch.arange(features.shape[0]), hard_negatives]
    
    else:
        psim, nsim = None, None

    return psim, nsim
