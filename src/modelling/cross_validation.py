# src/modelling/cross_validation.py

# Functions for k-fold expanding window cross-validation

from .train import train_hierarchical

from typing import Any, List, Dict, Tuple
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import SequentialSampler


def calc_means_unequal_lists(lists: List[List[float]]) -> List[float]:
    """
    Calculates the means of lists with unequal lengths.
    This is useful for calculating the average train and
    validation losses per epoch, which may have "early
    stopped" at different epoch moments. A note here is
    that values later on have, as a result, less weight

    :param lists: list of lists with floats
    :return: list of floats
    """                                    
    max_len = max([len(list) for list in lists])

    for list in lists:                  # pad with NaNs to equal length for
        if len(list) < max_len:         # compatibility with np.nanmean
            list.extend([np.nan] * (max_len - len(list)))
    return np.nanmean(lists, axis = 0)  # over 0-axis, so per epoch


def get_idx_k_fold_cross_validation_expanding_window(
        fold: int, n_folds: int, dataset_len: int
    ) -> Tuple[List[int], List[int]]:
    """
    Calculates the indices of expanding window k-fold cross validation by:
    - determining the fold size (= length of dataset divided by number of folds)
    - determining the ending indices of:
        - the training set (all data up to the current fold)
        - the validation set (all data in the current fold)
    - returning the training and validation indices
    
    :param fold: current fold
    :param n_folds: total number of folds
    :param dataset_len: length of the dataset
    :return: tuple of lists with training and validation indices
    """
    fold_size = dataset_len // n_folds # integer division
                                       # determine ending indices of:
    if fold == n_folds - 1:            # last fold
        train_end_idx = fold_size * fold
        val_end_idx = dataset_len
    else:                              # all other folds
        train_end_idx = fold_size * (fold + 1)
        val_end_idx = fold_size * (fold + 2)

    train_indices = list(range(0, train_end_idx))
    val_indices = list(range(train_end_idx, val_end_idx))
    return train_indices, val_indices


def get_idx_k_fold_cross_validation_sliding_window(
        fold: int, n_folds: int, dataset_len: int
    ) -> Tuple[List[int], List[int]]:
    """
    Another variation of k-fold cross validation, this time
    with a sliding window. It did not work optimally, but
    maybe useful when a big decrease in computation is needed.
    It works by sliding a training and validation window
    over the here implicit training set, given the dataset length
    and the fold, and how many folds to divide the dataset in

    :param fold: current fold
    :param n_folds: total number of folds
    :param dataset_len: length of the dataset
    :return: tuple of lists with training and validation indices
    """                             
    fold_size = dataset_len // (n_folds + 1)
    remainder = dataset_len % (n_folds + 1)

    train_start_idx = fold_size * fold + min(fold, remainder)
    train_end_idx = fold_size * (fold + 1) + min(fold, remainder)
    val_start_idx = train_end_idx
    val_end_idx = val_start_idx + fold_size

    train_indices = list(range(train_start_idx, train_end_idx))
    val_indices = list(range(val_start_idx, val_end_idx))
    return train_indices, val_indices
    

def k_fold_cross_validation_expanding_hierarchical(
        hp: Dict[str, Any], train_dataset, verbose = True
    ): 
    """
    Does k-fold expanding window cross validation training on a
    given model:
    - for each fold:
        - get the training and validation indices
        - create the train and validation loaders
        - train the model on the current fold
        - store the final validation loss
    - return the average of the final validation losses
    """
    val_losses_kfold = []

    for fold in range(hp['k_folds']):
        print(f"\n\tFold {fold + 1}/{hp['k_folds']}") if verbose else None
                                        # get indices for the current fold
        train_indices, val_indices = get_idx_k_fold_cross_validation_expanding_window(
            fold, hp['k_folds'], train_dataset.__len__())
                                        # create the train and validation loaders,
                                        # with random sampling for the train loader
        train_loader = DataLoader(train_dataset, batch_size = hp['batch_sz'], 
                            sampler = SubsetRandomSampler(train_indices))
        val_loader = DataLoader(train_dataset, batch_size = hp['batch_sz'], 
                            sampler = SequentialSampler(val_indices))
                                        # train new model on the current fold
        _, _, val_losses, _, _ = train_hierarchical(hp, train_loader, val_loader, verbose)
        val_losses_kfold.append(val_losses)

    return np.mean([losses[-1] for losses in val_losses_kfold])