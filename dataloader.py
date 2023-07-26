from torch.utils.data import Dataset, Sampler

import pandas as pd
import numpy as np

import os




class ADNIDataset(Dataset):
    def __init__(self, annotations_file, data_dir):
        self.data_labels = pd.read_csv(os.path.join('datasets', annotations_file+'.csv'))
        self.data_dir = data_dir
        self.label_mapping = {'CN': 0, 'AD': 1, 'LMCI': 2}

    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, index):
        # get the label of the subject
        label = self.label_mapping[self.data_labels.iloc[index, 1]]
        # get the mmse of the subject
        mmse_val = self.data_labels.iloc[index, 2]
        # get the mri
        data_path = os.path.join(self.data_dir,
                                self.data_labels.iloc[index, 0]+'.npy')
        data = np.load(data_path).astype(np.float32)
        data = np.expand_dims(data, axis=0)

        return data, label, np.asarray(mmse_val).astype(np.float32), index



class ADNISampler(Sampler):
    def __init__(self, 
                 sample_idx,
                 data_source='../input/cassava-leaf-disease-classification/train.csv'):
        super().__init__(data_source)
        self.sample_idx = sample_idx
        self.df_images = pd.read_csv(data_source)
        
    def __iter__(self):
        image_ids = self.df_images['image_id'].loc[self.sample_idx]
        return iter(image_ids)
    
    def __len__(self):
        return len(self.sample_idx)