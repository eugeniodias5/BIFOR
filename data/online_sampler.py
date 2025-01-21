import math, copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

class OnlineBackgroundSampler(Sampler):
    '''
    Sampler that forms batches of samples based on the background similarity.
    
    This sampler divides the dataset into clusters such that each cluster contains identities that fill up a batch evenly.
    Any remaining identities that cannot fill a batch are grouped together in a single batch.

    Args:
        dataset (list): Contains tuples of (img_tensor, id).
        batch_size (int): Total number of images in a batch.
        num_instances (int): Number of images per identity in a batch.
        model: Model f(1) used to calculate embeddings based on the background.
        device: Device to run the model.
        num_workers: Number of workers for loading the data.
    '''

    def __init__(self, dataset, batch_size, num_instances, model, device='cpu', num_workers=8):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less than num_instances={}'.format(batch_size, num_instances))
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.model = model
        self.device = device
        self.num_workers = num_workers

        self.num_ids_per_batch = batch_size // num_instances

        self.index_dic = defaultdict(list)
        for index, items in enumerate(dataset):
            id = items[1]
            self.index_dic[id].append(index)

        total_ids = len(self.index_dic)  # The total number of unique identities
        num_samples = len(dataset)
        self.id_num_imgs = math.ceil(num_samples / (total_ids * num_instances)) * num_instances

        # Adjust image indices per id for uniform number of images
        for id in self.index_dic:
            if len(self.index_dic[id]) < self.id_num_imgs:
                # Oversampling if there are fewer images than needed
                self.index_dic[id] = np.random.choice(self.index_dic[id], self.id_num_imgs, replace=True).tolist()
            else:
                # Undersampling if there are more images than needed
                self.index_dic[id] = random.sample(self.index_dic[id], self.id_num_imgs)

        self.ids = list(self.index_dic.keys())

        num_selected_ids = (total_ids // self.num_ids_per_batch) * self.num_ids_per_batch
        remaining_ids_count = total_ids % self.num_ids_per_batch

        selected_ids = random.sample(self.ids, num_selected_ids)
        remaining_ids = [id for id in self.ids if id not in selected_ids]

        self.index_dic = {id: self.index_dic[id] for id in selected_ids}
        self.ids = selected_ids

        self.num_clusters = len(selected_ids) // self.num_ids_per_batch

        self.mean_embeddings = self._calculate_mean_embeddings()
        self.cluster_assignments = self._constrained_kmeans(self.mean_embeddings)

        self.batches = self._form_batches()

        if remaining_ids_count > 0:
            self._add_remaining_id_batches(remaining_ids)

    def _calculate_mean_embeddings(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        embeddings_dict = defaultdict(list)
        
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating embeddings with the BackgroundNet"):
                images, ids = batch
                images = images.to(self.device)
                embeddings = self.model(images).cpu().numpy()
                for embedding, id in zip(embeddings, ids):
                    embeddings_dict[id.item()].append(embedding)

        mean_embeddings = [np.mean(embeddings_dict[id], axis=0) for id in self.ids]
        return np.vstack(mean_embeddings)

    def _constrained_kmeans(self, embeddings):
        print("Performing constrained K-means clustering...")
        kmeans = KMeansConstrained(
            n_clusters=self.num_clusters,
            size_min=self.num_ids_per_batch,
            size_max=self.num_ids_per_batch,
            random_state=42
        )
        return kmeans.fit_predict(embeddings)

    def _form_batches(self):
        batch_idxs_dict = defaultdict(list)
        for id in self.ids:
            idxs = copy.deepcopy(self.index_dic[id])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[id].append(batch_idxs)
                    batch_idxs = []

        final_idxs = []
        cluster_id_dict = defaultdict(list)
        for id, cluster in zip(self.ids, self.cluster_assignments):
            cluster_id_dict[cluster].append(id)

        while len(self.ids) >= self.num_ids_per_batch:
            for cluster in range(self.num_clusters):
                selected_ids = cluster_id_dict[cluster][:self.num_ids_per_batch]
                for id in selected_ids:
                    batch_idxs = batch_idxs_dict[id].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[id]) == 0:
                        self.ids.remove(id)
                        cluster_id_dict[cluster].remove(id)
        
        return final_idxs


    def _add_remaining_id_batches(self, remaining_ids):
        remaining_idxs = [idx for id in remaining_ids for idx in self.index_dic[id]]
        random.shuffle(remaining_idxs)
        # If there aren't enough indices to form a full batch, use what is available
        if len(remaining_idxs) < self.batch_size:
            self.batches.append(remaining_idxs)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) 
        
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
