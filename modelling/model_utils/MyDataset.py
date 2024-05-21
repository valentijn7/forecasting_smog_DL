import torch as tc
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, u, y):
        """Initliaze the dataset"""
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