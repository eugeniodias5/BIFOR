import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class BackgroundSampler(Sampler):
    """Sampler that finds a set of negative samples for each identity in a batch based on the background similarity.
       Used for training the model under offline triplet loss.

    Args:
        dataset (list): Contains tuples of (img_tensor, id).
        batch_size (int): Total number of images in a batch.
        model (torch.nn.Module): Neural model used to calculate embeddings.
        device (str): Device (e.g., 'cuda' or 'cpu') to run the model on.
    """

    def __init__(self, dataset, batch_size, model, device='cpu', num_workers=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = model
        self.device = device
        self.num_workers = num_workers
        
        # Precompute the embeddings and create the positive and negative sets
        self.embeddings, self.labels = self._compute_embeddings()

        print("Forming batches...")
        self.positive_sets = self._create_positive_sets()
        self.negative_sets = self._create_negative_sets()
        self.batches = self._form_batches()

    def _compute_embeddings(self):
        """Compute embeddings for all images in the dataset using the provided model."""
        self.model.eval()
        embeddings = []
        labels = []
        
        ds_dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        with torch.no_grad():
            for imgs, lbls in tqdm(ds_dl, desc="Calculating embeddings with the BackgroundNet"):
                imgs = imgs.to(self.device)
                embedding = self.model(imgs)  # Assumes model returns the embedding
                for emb, lbl in zip(embedding, lbls):
                    embeddings.append(emb.cpu().numpy())
                    labels.append(lbl)

        embeddings = np.vstack(embeddings)
        return embeddings, np.array(labels)
    
    def _create_positive_sets(self):
        """Create a set of positive samples for each identity."""
        positive_sets = {}
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            positive_indices = np.where(self.labels == label)[0]
            positive_sets[label] = list(positive_indices)
        
        return positive_sets
    
    def _create_negative_sets(self):
        """Create a predefined set of negative samples for each identity based on similarity."""
        negative_sets = {}
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            # Get embeddings of all images of the current identity (label)
            positive_mask = self.labels == label
            positive_embeddings = self.embeddings[positive_mask]
            
            # Get embeddings of all images of different identities (negatives)
            negative_mask = self.labels != label
            negative_embeddings = self.embeddings[negative_mask]
            negative_indices = np.where(negative_mask)[0]
            negative_labels = self.labels[negative_mask]
            
            # Calculate similarity between positive and negative embeddings
            similarity_matrix = cosine_similarity(positive_embeddings, negative_embeddings)
            
            # Sort the negative embeddings by similarity (descending order)
            sorted_negatives = np.argsort(-similarity_matrix, axis=1)
            
            # Store unique negative samples with different identities
            chosen_negatives = set()
            for row in sorted_negatives:
                for neg_idx in row:
                    neg_label = negative_labels[neg_idx]
                    if neg_label not in chosen_negatives:
                        chosen_negatives.add(neg_label)
                        negative_sets.setdefault(label, []).append(negative_indices[neg_idx])
                        break

                if len(negative_sets[label]) >= len(positive_embeddings):
                    break
        
        return negative_sets

    def _form_batches(self):
        """Precompute the batches array, containing anchors, positives and negatives."""
        batches = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            positive_indices = self.positive_sets[label]
            negative_indices = self.negative_sets[label]
            
            for anchor_idx in positive_indices:
                batches.append((anchor_idx, positive_indices, negative_indices))
        
        random.shuffle(batches)  # Shuffle the batches for randomness
        return batches

    def __iter__(self):
        batch = []
        
        for anchor_idx, positive_list, negative_list in self.batches:
            # Randomly select one positive and one negative
            positive_idx = random.choice([idx for idx in positive_list if idx != anchor_idx])
            negative_idx = random.choice(negative_list)
            
            # Append to batch
            batch.append((anchor_idx, positive_idx, negative_idx))
            
        random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return len(self.batches)


class OfflineSampler:
    """Sampler that finds a set of negative samples for each identity in a batch based on random selection.
       Used for training the model under offline triplet loss without using embeddings.

    Args:
        dataset (list): Contains tuples of (img_tensor, id).
        batch_size (int): Total number of images in a batch.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = dataset.targets
        self.batches = self._form_batches()

    def _form_batches(self):
        """Create batches directly without storing positive or negative sets."""
        positive_sets = {}
        negative_sets = {}
        unique_labels = set(self.labels)

        # Populate positive and negative sets using indices
        for label in unique_labels:
            positive_sets[label] = [i for i, lbl in enumerate(self.labels) if lbl == label]
            negative_sets[label] = [i for i, lbl in enumerate(self.labels) if lbl != label]
            random.shuffle(negative_sets[label])

        batches = []
        # Create batches using the sets
        for label, pos_indices in positive_sets.items():
            neg_indices = negative_sets[label]
            
            for anchor_idx in pos_indices:
                positive_candidates = [idx for idx in pos_indices if idx != anchor_idx]
                if positive_candidates and neg_indices:
                    positive_idx = random.choice(positive_candidates)
                    negative_idx = random.choice(neg_indices)
                    batches.append((anchor_idx, positive_idx, negative_idx))

        random.shuffle(batches)
        return batches

    def __iter__(self):
        """Return an iterator over precomputed batches."""
        return iter(self.batches)

    def __len__(self):
        """Return the number of precomputed batches."""
        return len(self.batches)
    