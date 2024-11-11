# src/modelling/MyDataset.py

# Comment: this class is not used in the current implementation of the project

import torch as tc
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    A basic custom dataset class for PyTorch, which takes input (u) and output (y) data
    and allows easy querying and, for example, compatibility with PyTorch's DataLoader
    """
    def __init__(self, u, y):
        """
        Initialize the dataset with the input (u) and output (y) data,
        and the number of samples (n), input features (u_f) and output features (y_f)
        """
        self.u = tc.tensor(u.values, dtype = tc.float32)
        self.y = tc.tensor(y.values, dtype = tc.float32)
        self.n = u.shape[0]
        self.u_f = u.shape[1]
        self.y_f = y.shape[1]

    def __len__(self):
        """Return the length of the dataset"""
        return self.n
    
    def __features_in__(self):
        """Return the number of input features"""
        return self.u_f
    
    def __features_out__(self):
        """Return the number of output features"""
        return self.y_f
    
    def __getitem__(self, idx):
        """Return the idx-th pair of the dataset"""
        return self.u[idx], self.y[idx]