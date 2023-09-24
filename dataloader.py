from torch.utils.data import Dataset, Sampler

import pandas as pd
import numpy as np

import os




class ADNIDataset(Dataset):
    def __init__(self, annotations_file, data_dir):
        self.data_labels = pd.read_csv(os.path.join('datasets', annotations_file+'.csv'))
        self.data_dir = data_dir
        # self.label_mapping = {'CN': 0, 'AD': 1, 'LMCI': 2}
        self.label_mapping = self.get_classes()


    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, index):
        # get the label of the subject
        label = self.label_mapping.index(self.data_labels.iloc[index, 1])
        # get the mmse of the subject
        mmse_val = self.data_labels.iloc[index, 2]
        # get the mri
        data_path = os.path.join(self.data_dir,
                                self.data_labels.iloc[index, 0]+'.npy')
        data = np.load(data_path).astype(np.float32)
        data = np.expand_dims(data, axis=0)

        return data, label, np.asarray(mmse_val).astype(np.float32)
    
    def get_classes(self):
        # returns a list containing the 2 class labels sorted by alphabetical order
        res = self.data_labels['status'].value_counts().to_dict()
        ordered_labels_list = [lab for lab in res.keys()]
        ordered_labels_list.sort()
        return ordered_labels_list

    def get_class_imbalance_ratio(self):
        # returns the ratio between the two classes
        count0 = float(self.data_labels.value_counts('status')[self.label_mapping[0]])
        count1 = float(self.data_labels.value_counts('status')[self.label_mapping[1]])
        return count0 / count1
    

if __name__ == '__main__':
    # useful to debug the dataloader
    dataset = ADNIDataset("ADNI1_prob_full_data", "/mnt/mydrive/Matteo/data_adni1_adni2_smooth/")
