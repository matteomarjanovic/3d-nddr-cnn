import torch

from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import ADNIDataset
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import json
import argparse
import os


def main(config:dict):
    dataset_df = pd.read_csv(os.path.join('datasets', config['dataset']+'.csv'))
    dataset = ADNIDataset(config['dataset'], config['data_dir'])

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=config['n_folds'],
                            shuffle=True,
                            random_state=42)

    index_list = list(dataset_df.index.values)
    label_list = dataset_df['status']
    folds_to_execute = config['folds']
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(index_list, label_list)):
        if fold not in folds_to_execute:
            continue
        train_dataloader = DataLoader(dataset,
                                      batch_size=config['cnn']['batch_size'],
                                      sampler = SubsetRandomSampler(train_ids))
        valid_dataloader = DataLoader(dataset,
                                      batch_size=config['cnn']['batch_size'],
                                      sampler = SubsetRandomSampler(valid_ids))










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train the model.')
    parser.add_argument('-c',
                        '--config',
                        help='Configuration for the training of the model.')
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, 'r') as config_f:
        config = json.load(config_f)
    main(config)