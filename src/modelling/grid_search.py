# src/modelling/grid_search.py

# Functions for grid search

from .cross_validation import k_fold_cross_validation_expanding_hierarchical

from typing import Any, List, Dict, Tuple
import numpy as np
import torch


def filter_dict_by_keys(
        dict: Dict[str, Any], secondary_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Filters a dictionary by the keys of another dictionary. For example:
    dict = {'a': 1, 'b': 2, 'c': 3} and secondary_dict = {'a': 0, 'c': 0}
    returns {'a': 1, 'c': 3}

    :param dict: dictionary to be filtered
    :param secondary_dict: dictionary with keys to filter by
    :return: filtered dictionary
    """
    return {k: dict[k] for k in secondary_dict if k in dict}


def update_dict(
        dict: Dict[str, Any] , secondary_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
    """
    Updates a dictionary with the keys of another dictionary. For example:
    dict = {'a': 1, 'b': 2, 'c': 3} and secondary_dict = {'a': 0, 'c': 0}
    returns {'a': 0, 'b': 2, 'c': 0}

    :param dict: dictionary to be updated
    :param secondary_dict: dictionary with keys to update with
    :return: updated dictionary
    """
    return {**dict, **secondary_dict}


def print_dict_vertically(d: Dict[str, Any]) -> None:
    """
    Prettyprints a dictionary formatted vertically
    
    :param d: dictionary to be printed
    """
    max_key_len = max(len(key) for key in d.keys())
    for key, value in d.items():
        print(f"{key:{max_key_len}}: {value}")


def print_dict_vertically_root(d: Dict[str, Any]) -> None:
    """
    Pretty prints a dictionary formatted vertically,
    but takes the square root of the values (useful for
    printing RMSEs, for example)

    :param d: dictionary to be printed
    """
    max_key_len = max(len(key) for key in d.keys())
    for key, value in d.items():
        print(f"{key:{max_key_len}}: {np.sqrt(value)}")


def ensure_integers(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prevents a nasty occassional type error, or rather,
    untrackable bug, by ensuring that the hidden_layers and
    hidden_units are integers
    
    :param configs: list of dictionaries with hyperparameters
    :return: list of dictionaries with hyperparameters
    """
    for config in configs:
        config['hidden_layers'] = int(config['hidden_layers'])
        config['hidden_units'] = int(config['hidden_units'])
    return configs


def gen_configs(hp_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Helper function to initiate the configuration generator
    # https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists
    
    :param hp_space: dictionary with hyperparameter space
    :return: list of dictionaries with hyperparameters
    """
    keys = list(hp_space.keys())
    values = list(hp_space.values())
    configs = [dict(zip(keys, combi))
                for combi in np.array(np.meshgrid(*values)).T.reshape(-1, len(values))]
    return ensure_integers(configs)


def print_current_config(config: Dict[str, Any]) -> None:
    """
    Prints the current configuration
    
    :param config: dictionary with hyperparameters
    """
    print("CURRENT CONFIGURATION:")
    print_dict_vertically(config)
    print()


def print_end_of_grid_search(best_hp: Dict[str, Any], best_val_loss: float) -> None:
    """
    Prints the results of the grid search
    
    :param best_hp: dictionary with best hyperparameters
    :param best_val_loss: best validation loss
    """
    print(f"##### Best average validation loss so far: {best_val_loss:.6f} #####")
    print("With configuration:")
    print_dict_vertically(best_hp)
    print()


def grid_search(
        hp: Dict[str, Any], hp_space: Dict[str, Any],
        train_dataset: torch.utils.data.Dataset, verbose: bool = False
    ) -> Tuple[Dict[str, Any], float]:
    """
    Perform an ordinary grid search through hyperparameter space:
    - for each configuration in the hyperparameter space:
        - train a model
        - calculate the average validation loss
        - if the average validation loss is lower than the best so far:
            - update the best hyperparameters and validation loss
    - return best hyperparameters and validation loss
    The procedure is explained in more detail in Section 3.3.2 of thesis
    
    :param hp: dictionary with default hyperparameters
    :param hp_space: dictionary with hyperparameter space
    :param train_dataset: training dataset
    :param verbose: whether to print the results of each configuration
    :return: best hyperparameters and validation loss
    """
    best_hp, best_val_loss = {}, np.inf
                                        
    for config in gen_configs(hp_space):
        config_dict = update_dict(hp, config)
        if verbose:
            print_current_config(config_dict)
                                        
        current_config_loss = k_fold_cross_validation_expanding_hierarchical(
            config_dict, train_dataset, verbose
        )
        if current_config_loss < best_val_loss:
            best_hp = config_dict.copy()
            best_val_loss = current_config_loss
        if verbose:
            print(f"Average final validation loss: {current_config_loss:.6f}")
            print_end_of_grid_search(best_hp, best_val_loss)

    if not verbose:                     
        print_end_of_grid_search(best_hp, best_val_loss)
    
    return best_hp, best_val_loss