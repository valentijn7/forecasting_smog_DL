# src/modelling/plots.py

# Plots for the modelling package, including but not limited to:
# - plots to transform the data to proper format

from .denormalise import normalise_linear_inv
from .denormalise import denormalise

from typing import Any, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables for plots.py (bad but coincidentally useful practice!)
CONTAMINANTS = ['NO2', 'O3', 'PM10', 'PM25']
MINMAX_PATH = "../data/data_combined/contaminant_minmax.csv"


def set_style() -> None:
    """
    Sets an easy-to-read standard styles for plots in notebooks.
    For paper plots, use more explicit plotting functions with
    e.g. LaTeX labels and white backgrounds
    """
    sns.set_style("darkgrid")
    sns.set_palette("dark")
    sns.set_context("notebook")


def get_pred_and_gt(
        model: Any, dataset: torch.utils.data.Dataset, idx: int, denorm = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the predictions and ground truth for a given row
    in the dataset, optionally denormalising the values. This
    is particularly useful for plotting the results for a
    specific subset of the dataset
    
    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param idx: index of the row in the dataset
    :param denorm: whether to denormalise the values (default: True)
    :return: tuple of predictions and ground truth, as numpy arrays
    """                                 # with .unsqueeze(0) we add a batch dimension
    input_data = dataset[idx][0].unsqueeze(0)
    ground_truth = dataset[idx][1].unsqueeze(0)
    
    if denorm:                          # denormalise if necessary
                                        # concatenate the model output along the second dimension,
                                        # which contains the contaminant values--the model outputs
                                        # a tensor with shape (batch_size, time_steps, n_contaminants),
                                        # or, more generally, (batch_size, time_steps, n_features)
        pred = denormalise(torch.cat(model(input_data), dim = 2),
                           MINMAX_PATH).detach().numpy()
        gt = denormalise(ground_truth, MINMAX_PATH).detach().numpy()
    else:
        pred = torch.cat(model(input_data), dim = 2).detach().numpy()
        gt = ground_truth.detach().numpy()

    if pred.shape[0] != gt.shape[0]:
        raise ValueError("get_pred_and_gt(): pred and gt should be comparable")
    return pred, gt


def get_index(list: list, component: str) -> int:
    """
    Small helper function that returns the index of
    the component in the list, useful for finding the
    index of a contaminant in the list of contaminants

    :param list: list to be searched
    :param component: component to find in the list
    :return: index of the component in the list
    """
    try:
        return list.index(component)
    except ValueError:
        return "Component not found in the list"
    

def choose_plot_component_values(
        model: Any, dataset: torch.utils.data.Dataset, idx: int, comp: str, denorm = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chooses the correct component values for plotting by:
    - getting the predictions and ground truth for the given index
    - finding the index of the component in the list of contaminants
    - returning the values for the component in the predictions
      and ground truth

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param idx: index of the row in the dataset
    :param comp: component to be plotted
    :param denorm: whether to denormalise the values (default: True)
    :return: tuple of predictions and ground truth for the component
    """
    pred, gt = get_pred_and_gt(model, dataset, idx, denorm)
    comp_idx = get_index(CONTAMINANTS, comp)
    return np.squeeze(pred[:, :, comp_idx]), np.squeeze(gt[:, :, comp_idx])


def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
    """
    Plots the train and validation losses (risk vs empirical risk)

    :param train_losses: list of training losses
    :param val_losses: list of validation losses
    """
    set_style()

    sns.lineplot(x = range(len(train_losses)), y = train_losses, label = "Rtrain")
    sns.lineplot(x = range(len(val_losses)), y = val_losses, label = "Rvalidation")

    plt.title('Empirical risk "training error" vs Risk "validation/testing error"')
    plt.xlabel("epoch")
    plt.ylabel("risk / empirical risk")
    plt.show()


def plot_losses_normalised(
        losses_1: List[float], losses_2: List[float], what: str = ""
    ):
    """
    Plots two sequences of losses, normalised by themselves--useful
    for comparing two different loss sequences with different ranges

    :param losses_1: list of losses 1
    :param losses_2: list of losses 2
    :param what: what the losses represent (e.g. "train" or "val") or
                 any other string to be added to the title    
    """
    set_style()

    sns.lineplot(x = range(len(losses_1)),
                 y = normalise_linear_inv(
                     torch.tensor(losses_1), min(losses_1), max(losses_1)
                 ), label = "Losses 1")
    sns.lineplot(x = range(len(losses_2)),
                 y = normalise_linear_inv(
                     torch.tensor(losses_2), min(losses_2), max(losses_2)
                 ), label = "Losses 2")

    plt.title(f'Losses 1 vs Losses 2 - {what}')
    plt.xlabel("epoch")
    plt.ylabel("loss (normalised)")
    plt.show()


def plot_flexibility(
        empirical_risk: List[float], risk: List[float], x_labels: List[str]
    ) -> None:
    """
    Plots risk over model flexbility. Only useful after lots
    of calculations, so not the most used function
    
    :param empirical_risk: list of empirical risks
    :param risk: list of risks
    :param x_labels: labels for the x-axis
    """
    set_style()

    sns.scatterplot(x = range(len(empirical_risk)), y = empirical_risk, label = "R_emp")
    sns.scatterplot(x = range(len(risk)), y = risk, label = "R")

    plt.title('Empirical risk/Risk vs Model Flexibility')
    plt.xticks(range(len(x_labels)), x_labels)
    plt.xlabel("model flexibility")
    plt.ylabel("risk / empirical risk")
    plt.show()


def plot_pred_vs_gt(
        model: Any, dataset: torch.utils.data.Dataset, row: int, comp: str, n_hours_y: int
    ) -> None:
    """
    Plots predictions (dotted) vs ground truth (solid) with basic lay-out

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param row: index of the row in the dataset
    :param comp: component to be plotted
    :param n_hours_y: number of hours in the output sequence (N_HOURS_Y)
    """
    pred, gt = choose_plot_component_values(model, dataset, row, comp)
    
    set_style()
    
    sns.lineplot(x = range(n_hours_y), y = pred, label = f"{comp}_pred", linestyle = 'dashed')
    sns.lineplot(x = range(n_hours_y), y = gt, label = f"{comp}_true")
    
    plt.title(f"{comp} prediction vs ground truth")
    plt.xlabel("time in hrs")
    plt.ylabel(f"{comp} concentration")
    plt.show()