# src/modelling/TimeSeriesDataset.py

# Custom PyTorch dataset tailored for the timeseries problem. In short,
# it takes all the data created by the pipeline, processes them to
# (u, y)-pairs in its constructor, stores them as data members, and
# returns them when indexed (for example through a DataLoader).
#   It handles:
# - varying numbers of input- and output features;
# - varying sequence- and output lengths; and
# - varying numbers of input- and output frames.
# The last point solves the problem of the discontinuous sampling range
# (i.e. data of just a few months per year is used, and not full years).
# It does so by just creating the pairs for each separate dataframe and
# then concatenating them (which is all done in the constructor). This
# dataset also precomputes all the input-output pairs, which takes some
# memory for the sake of faster indexing (during e.g. training).
#   Thesis Section 3.2 explains the sampling procedure a bit more formally
# and in more detail.
#   Overview of the methods:
# - __init__(): Constructor; noteworthy here is the len_step parameter, which
#               is the step between samples (step taken between the pairs during
#               the sampling procedure). To increase computational efficiency,
#               increase it, but it might decrease the model's performance.
#               Increase it by a lot if just testing if the code or a model works
#               properly, as it will save a lot of time. The thesis uses a step
#               of 24 I believe, which goes to show that the amount of data (at
#               least in previous experiments) shows diminishing returns.
# - _precompute_pairs__(): Helper function to the constructor; it precomputes
#                          n samples from the input and output dataframes
# - get_full_u_sequence(): Returns the entire sequence from u as a single tensor
# - get_full_y_sequence(): Returns the entire sequence from y as a single tensor
# - __len_seq__(): Returns the length of the input sequence (72 hours in the thesis)
# - __n_features_in__(): Returns the number of input features (10 in the thesis)
# - __n_features_out__(): Returns the number of output features (4 pollutants)
# - __len__(): Returns the number of possible sampling points (the number of pairs)
# - __getitem__(): Returns the idx-th pair of the dataset (input-output pair,
#                  both as tensors, so this is a frequently used method)

from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Customized PyTorch dataset for the timeseries problem, and probably more
    timeseries problems. It takes all the data created by the pipeline, processes
    them to (u, y)-pairs in its constructor, stores them as data members, and
    returns them when indexed (for example through a DataLoader).

    Methods:
    - __init__(): Constructor
    - _precompute_pairs__(): Precomputes n samples from the input and output dataframes
    - get_full_u_sequence(): Returns the entire sequence from u as a single tensor
    - get_full_y_sequence(): Returns the entire sequence from y as a single tensor
    - __len_seq__(): Returns the length of the input sequence
    - __n_features_in__(): Returns the number of input features
    - __n_features_out__(): Returns the number of output features
    - __len__(): Returns the number of possible sampling points
    - __getitem__(): Returns the idx-th pair of the dataset
    """
    def __init__(
            self,
            u: List[pd.DataFrame],
            y: List[pd.DataFrame],
            num_dfs: int,
            len_seq: int,
            len_out: int,
            len_step: int):
        """
        Constructor of the TimeSeriesDataset class. It takes the input and output
        dataframes, the number of dataframes, the length of the input sequence, the
        length of the output sequence, and the step between samples. It initializes
        the data members and precomputes the input-output pairs.

        :param u: list of input dataframes
        :param y: list of output dataframes
        :param num_dfs: number of dataframes (in hindsight, I cannot remember why
                        this is a parameter, but it at least creates a conscious
                        check of what is fed into the constructor... so let's keep it)
        :param len_seq: length of the input sequence (amount of hours, or just timesteps
                        in general as the sampling rate is hourly)
        :param len_out: length of the output sequence
        :param len_step: step between samples; increase for computational efficiency
        """     
        if num_dfs < 1:
            raise ValueError("Invalid number of dataframes")
        if isinstance(u, pd.DataFrame): # convert to list if necessary,
            u = [u]                     # in case of single dataframe
        if isinstance(y, pd.DataFrame):
            y = [y]
                                        # init data members:
        self.len_seq = int(len_seq)     # length of: the input sequence
        self.len_out = int(len_out)     #            the output sequence
        self.len_step = int(len_step)   #            the step between samples
        self.u = u                      # lists of initial input dataframes
        self.y = y
        self.u_f = u[0].shape[1]        # number of: input features
        self.y_f = y[0].shape[1]        #            output features
                                        #            samples
        self.n = sum(df.shape[0] for df in self.u)
                                        # precompute input-output pairs:
        self.n_samp = 0                 # pair counter/list set to zero
        self.pairs = []
        for u_df, y_df in zip(self.u, self.y):
            n = u_df.shape[0]           # number of time samples for the current pair
                                        # number of possible sampling points for the current pair
            n_samp = (n - self.len_seq - self.len_out + 1) // self.len_step
            self.n_samp += n_samp       # add to total number of possible sampling points
                                        # precompute input-output pairs for the current pair
            self.pairs.extend(self._precompute_pairs__(u_df, y_df, n_samp))

    def _precompute_pairs__(
            self,
            u: pd.DataFrame,
            y: pd.DataFrame,
            n_samp: int
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Precomputes n samples from the input and output dataframes
        
        :param u: input dataframe
        :param y: output dataframe
        :param n_samp: number of samples to precompute
        :return: list of input-output pairs
        """
        pairs = []
        # Loop over the dedicated sampling points, with steps of len_step
        for idx in range(0, n_samp * self.len_step, self.len_step):
            # Index the input and output sequences
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

    # Accessors:
    def __len_seq__(self):
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