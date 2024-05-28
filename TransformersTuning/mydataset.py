import torch
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, mode='train'):
        self.encodings = encodings
        if mode !="train":
            self.labels=  [0]*len(encodings)
        else: self.labels = labels


    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

