import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class LabeledHybrid(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        self.label_list_file = os.path.join(self.data_root, f'{self.subset}_num.txt')

        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'LabeledHybrid')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'LabeledHybrid')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        print_log(f'[DATASET] Open file {self.label_list_file}', logger = 'LabeledHybrid')
        with open(self.label_list_file, 'r') as f:
            lines_label = f.readlines()

        self.file_list = []
        for line in lines:
            self.file_list.append(line.strip())
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'LabeledHybrid')
        self.label_list = []
        for line_label in lines_label:
            self.label_list.append(np.array(int(line_label.strip())))
        print_log(f'[DATASET] {len(self.label_list)} labels were loaded', logger = 'LabeledHybrid')


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        label = self.label_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample)).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return 'LabeledHybrid', 'sample', (data, label)

    def __len__(self):
        return len(self.file_list)