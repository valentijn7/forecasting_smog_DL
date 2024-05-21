import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Customized for the time series problem. It handles:
    - varying numbers of input- and output features;
    - varying sequence- and output lengths; and
    - varying numbers of input- and output frames.
    The last point solves the problem of the discontinuous sampling range
    of the data. It does so by just creating the pairs for each separate 
    dataframe and then concatenating them. This is done in the constructor.

    This dataset also precomputes all the input-output pairs, which makes
    takes some memory for the sake of faster indexing (during e.g. training).
    """
    def __init__(self, u: list, y: list, num_dfs, len_seq: int, len_out: int, len_step: int):
        """Constructor"""     
        if num_dfs < 1:
            raise ValueError("Invalid number of dataframes")
        if isinstance(u, pd.DataFrame): # convert to list if necessary
            u = [u]
        if isinstance(y, pd.DataFrame):
            y = [y]
                                        # init data members:
        self.len_seq = int(len_seq)     # length of: the input sequence
        self.len_out = int(len_out)     #            the output sequence
        self.len_step = int(len_step)   #            the step between samples
        self.u = u                      # initial input dataframes
        self.y = y
        self.u_f = u[0].shape[1]        # number of: input features
        self.y_f = y[0].shape[1]        #            output features
                                        #            samples
        self.n = sum(df.shape[0] for df in self.u)
                                        # precompute input-output pairs:
        self.n_samp = 0
        self.pairs = []
        for u_df, y_df in zip(self.u, self.y):
            n = u_df.shape[0]           # number of time samples for the current pair
                                        # number of possible sampling points for the current pair
            n_samp = (n - self.len_seq - self.len_out + 1) // self.len_step
            self.n_samp += n_samp       # add to total number of possible sampling points
                                        # precompute input-output pairs for the current pair
            self.pairs.extend(self._precompute_pairs__(u_df, y_df, n_samp))

    def _precompute_pairs__(self, u: pd.DataFrame, y: pd.DataFrame, n_samp: int):
        """Precomputes n samples from the input and output dataframes"""
        pairs = []
        for idx in range(0, n_samp * self.len_step, self.len_step):
            seq_in = torch.tensor(
                u.iloc[idx : idx + self.len_seq].values,
            ).float()
            seq_out = torch.tensor(
                y.iloc[idx + self.len_seq - 23 : idx + self.len_seq - 23 + self.len_out].values,
            ).float()
            pairs.append((seq_in, seq_out))
        return pairs
    
    def get_full_u_sequence(self):
        """Returns the entire sequence from u as a single tensor"""
        if len(self.u) != 1:
            raise ValueError("get_full_u_sequence() can only be used when there is exactly one dataframe in u")
        return torch.tensor(self.u[0].values).float()

    def get_full_y_sequence(self):
        """Returns the entire sequence from y as a single tensor"""
        if len(self.y) != 1:
            raise ValueError("get_full_y_sequence() can only be used when there is exactly one dataframe in y")
        return torch.tensor(self.y[0].values).float()

    def __len_seq__(self):              # Accessors:
        """Returns the length of the input sequence"""
        return self.len_seq

    def __n_features_in__(self):
        """Returns the number of input features"""
        return self.u_f
    
    def __n_features_out__(self):
        """Returns the number of output features"""
        return self.y_f

    def __len__(self):
        """Returns the number of possible sampling points"""
        return self.n_samp
    
    def __getitem__(self, idx):
        """Returns the idx-th pair of the dataset"""
        return self.pairs[idx]