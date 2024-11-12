# src/modelling/test.py

# Functions that test the models' performance

from .denormalise import denormalise

from typing import Any, List, Dict
import numpy as np
import torch
import torch.nn as nn


def test_hierarchical(
        model: Any, loss_fn: Any, test_loader: nn.utils.data.DataLoader,
        denorm: bool = False, path: str = None
    ) -> float:
    """
    Evaluates an hierarchical model, a model with branches, on a test set.
    Or, more literally, it passes a DataLoader's batches through the model
    and calculates the average loss over the entire Dataset

    :param model: model to evaluate, must be some PyTorch type model
    :param loss_fn: loss function to use, PyTorch defined, or PyTorch inherited
    :param test_loader: DataLoader to get batches from
    :param denorm: whether to denormalise the data before calculating loss
    :param path: path to the file containing the minmax values for the data
    :return: average loss over the entire test set
    """                     
    model.eval()                            # set model to evaluation mode
    test_loss = np.float64(0)               # initialise test loss to 0

    with torch.no_grad():                   # don't calculate gradients
                                            # loop over all batches
        for batch_test_u, batch_test_y in test_loader:
                                            # pass batch through model
                                            # and concatenate the outputs
            pred = torch.cat(model(batch_test_u), dim = 2)
            
            if denorm:                      # if denorm is True, denormalise
                pred = denormalise(pred, path)
                batch_test_y = denormalise(batch_test_y, path)
                                            # calculate loss and add to test_loss
            test_loss += loss_fn(pred, batch_test_y).item()

    return test_loss / len(test_loader)     # return average loss over test set


def test_hierarchical_separately(
        model: Any, loss_fn: Any, test_loader: torch.utils.data.DataLoader,
        denorm: bool = False, path: str = None,
        components = ['NO2', 'O3', 'PM10', 'PM25']
    ) -> Dict[str, float]:
    """
    Similar to test_hierarchical(), but calculates the loss for each
    contaminant/pollutant/predictive variable separately and returns
    a dictionary with the results

    :param model: model to evaluate, must be some PyTorch type model
    :param loss_fn: loss function to use, PyTorch defined, or PyTorch inherited
    :param test_loader: DataLoader to get batches from
    :param denorm: whether to denormalise the data before calculating loss
    :param path: path to the file containing the minmax values for the data
    :param components: list of contaminant/pollutant/component names
    :return: dictionary with contaminant names as keys and losses as values
    """
    model.eval()                            # set model to evaluation mode
    test_losses = [np.float64(0) for _ in components]

    with torch.no_grad():                      # don't calculate gradients
                                            # loop over all batches
        for batch_test_u, batch_test_y in test_loader:
                                            # pass batch through model
                                            # and concatenate the outputs
            pred = torch.cat(model(batch_test_u), dim = 2)
            if denorm:                      # if denorm is True, denormalise
                pred = denormalise(pred, path)
                batch_test_y = denormalise(batch_test_y, path)
                                            # calculate loss and add to test_loss
                                            # for each component separately
            for comp in range(len(components)):                    
                test_losses[comp] += loss_fn(
                    pred[:, :, comp],       # take only the component of interest
                    batch_test_y[:, :, comp]
                ).item()                    # item() to get the actual value

    for comp in range(len(components)):     # divide by number of batches
        test_losses[comp] /= len(test_loader)
    return {comp: loss for comp, loss in zip(components, test_losses)}