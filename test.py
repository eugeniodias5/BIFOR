import time, os
from tqdm import tqdm

import numpy as np
import torch

from utils import compute_mean_features, evaluate_cmc_map, save_closest_imgs_json


def fliplr(img, device):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).to(device).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def test(model, dataloader, device, cfg, top_k=50):  
    '''
    Test function that receives the model, data loaders and returns the calculated metrics: cmc and map.

    Args:
        model: Model to evaluate.
        dataloader: Test dataloader.
        device: Device to run the model.
        cfg: Configuration dictionary (yaml file).
        top_k: Maximum k to calculate the top-k.

    Returns:
        test_res: Dictionary containing the calculated metrics (mAP and CMC).
    '''

    model.eval()

    test_res = {"map": 0.0, "cmc": []}

    features = torch.FloatTensor().to(device)
    labels = []
    ds_name = cfg['dataset_name']

    start_idx = 0

    since = time.time()
    with torch.no_grad():
        print('Testing...')
        print('-' * 10)
        for anchor, batch_labels in tqdm(dataloader, desc="Processing Test batch"):
            anchor = anchor.to(device)
            
            img_flip = fliplr(anchor, device)

            f = model(anchor)
            ff = f + model(img_flip)

            if ff.dim() == 3:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)

            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            batch_size = anchor.size(0)
            features = torch.cat((features, ff), 0)
            for sample_label in batch_labels: labels.append(int(sample_label))
            start_idx += batch_size


        labels = torch.tensor(labels)

        # If the dataset is the Sibetan, we compute the mean of the STSDs
        if ds_name == 'Sibetan':
            mean_features = compute_mean_features(features, labels)
            labels = dataloader.dataset.clus_labels
            clus_labels = torch.tensor(labels)

            test_res['map'], test_res['cmc'] = evaluate_cmc_map(mean_features.to('cpu'), clus_labels.to('cpu'), top_k)
        
        else:
            # Evaluate CMC and mAP
            test_res['map'], test_res['cmc'] = evaluate_cmc_map(features.to('cpu'), labels.to('cpu'), top_k) 

        time_elapsed = time.time() - since

        print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))        
        print(f"Test mAP: {test_res['map']}, top-1: {test_res['cmc'][0]}, top-5: {test_res['cmc'][4]}, top-10: {test_res['cmc'][9]}")    

        if cfg['save_closest_json']:
            # If it's true, we will save a json containing the images that are closest in embedding space
            img_paths = dataloader.dataset.samples

            json_path = os.path.join(cfg['save_dir'], 'closest_imgs.json')
            save_closest_imgs_json(features=features.to('cpu'), img_paths=img_paths, save_path=json_path)

    return test_res
