from torch.utils.data import Dataset,DataLoader
import numpy as np


class Datasetnpy(Dataset):
    def __init__(self):
        self.train_data= np.load('X_train.npy').reshape(-1,100)
    def __getitem__(self, item):
        real_data = self.train_data[item]
        return real_data
    def __len__(self):
       return self.train_data.shape[0]

