# src/modelling/plots.py

# Plots for the modelling package, including but not limited to:
# - plots to transform the data to proper format
# - plots to visualise the losses of the model
# - plots to visualise the predictions of the model
# - etc.

from .denormalise import normalise_linear_inv
from .denormalise import denormalise

from typing import Any, Tuple, List
import numpy as np
import pandas as pd
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


def set_thesis_style() -> None:
    """
    Sets a more precise style suited for the thesis
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "axes.labelsize": 8,     # fontsize for x and y labels
        "xtick.labelsize": 8,    # fontsize of the tick labels
        "ytick.labelsize": 8,    # fontsize of the tick labels
        "legend.fontsize": 8,    # fontsize of the legen
    })
    
    sns.set_theme(style = 'ticks')
    sns.set_context("paper")


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


def get_x_days_of_pred_and_gt(
        model: Any, dataset: torch.utils.data.Dataset, pair: int, comp: str,
        denormalise: bool = True, hierarchical: bool = False, days: int = 14
    ) -> None:
    """
    Similar to get_pred_and_gt(), helper function to get the predictions
    and ground truth for a given number of days, concatenating the results
    in a single DataFrame for each of the predictions and ground truth values.
    Useful for easily plotting them, for example

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param pair: index of the row in the dataset to start from
    :param comp: component to be plotted (always double check if this works OK)
    :param denormalise: whether to denormalise the values (default: True)
    :param hierarchical: whether the model is hierarchical (default: False)
    :param days: number of days to plot (default: 14, aka two weeks)
    """
    pred, gt = pd.DataFrame(), pd.DataFrame()
    for idx in range(days):
        pred_temp, gt_temp = choose_plot_component_values(model,
                                                          dataset,
                                                          pair + idx,
                                                          comp,
                                                          denormalise,
                                                          hierarchical)
        pred = pd.concat([pred, pd.DataFrame(pred_temp)], ignore_index = True)
        gt = pd.concat([gt, pd.DataFrame(gt_temp)], ignore_index = True)
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


def get_x_days_of_comp(
        model: Any, dataset: torch.utils.data.Dataset, pair: int, comp: str, 
        denormalise: bool = True, hierarchical: bool = False, days: int = 14
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function for calling choose_plot_component_values() for a
    given number of days, concatenating the results in a single DataFrame

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param pair: index of the row in the dataset to start from
    :param comp: component to be plotted
    :param denormalise: whether to denormalise the values (default: True)
    :param hierarchical: whether the model is hierarchical (default: False)
    :param days: number of days to plot (default: 14, aka two weeks)
    :return: tuple of predictions and ground truth for the component
    """
    pred, gt = pd.DataFrame(), pd.DataFrame()
    for idx in range(days):
        pred_temp, gt_temp = choose_plot_component_values(model,
                                                          dataset,
                                                          pair + idx,
                                                          comp,
                                                          denormalise,
                                                          hierarchical)
        pred = pd.concat([pred, pd.DataFrame(pred_temp)], ignore_index = True)
        gt = pd.concat([gt, pd.DataFrame(gt_temp)], ignore_index = True)
    return pred, gt


def get_all_data_of_all_comps(
        model: Any, dataset: torch.utils.data.Dataset,
        hierarchical: bool = False, denorm: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper function to get all predictions and ground truth values
    for all components, concatenating the results in a single DataFrame,
    does so by calling get_x_days_of_comp() for each component. The amount
    of batches is hardcoded to 93

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param hierarchical: whether the model is hierarchical (default: False)
    :param denorm: whether to denormalise the values (default: True)
    :return: tuple of DataFrames with predictions and ground truth for each component
    """
    df_NO2_pred, df_NO2_gt = get_x_days_of_comp(
        model, dataset, 0, 'NO2', denorm, hierarchical, 93
        )
    df_O3_pred, df_O3_gt = get_x_days_of_comp(
        model, dataset, 0, 'O3', denorm, hierarchical, 93
        )
    df_PM10_pred, df_PM10_gt = get_x_days_of_comp(
        model, dataset, 0, 'PM10', denorm, hierarchical, 93
        )
    df_PM25_pred, df_PM25_gt = get_x_days_of_comp(
        model, dataset, 0, 'PM25', denorm, hierarchical, 93
        )
    return df_NO2_pred, df_NO2_gt, df_O3_pred, df_O3_gt, \
        df_PM10_pred, df_PM10_gt, df_PM25_pred, df_PM25_gt


def get_all_model_predictions(
        model: Any, dataset: torch.utils.data.Dataset,
        hierarchical: bool = False, denorm: bool = False
    ) -> pd.DataFrame:
    """
    Gets all predictions of a model on a dataset into one df
    (only used for some specific plots, and, 92, the number
    or batches in the dataset is hardcoded in)

    :param model: model to be used for prediction
    :param dataset: dataset to be used for prediction
    :param hierarchical: whether the model is hierarchical (default: False)
    :param denorm: whether to denormalise the values (default: False)
    :return: DataFrame with all predictions
    """
    df_NO2 = pd.DataFrame()
    df_O3 = pd.DataFrame()
    df_PM10 = pd.DataFrame()
    df_PM25 = pd.DataFrame()

    for idx in range(0, 92):
        if hierarchical:
            pred = torch.cat(model(dataset[idx][0].unsqueeze(0)), dim = 2).squeeze(0)
        else:
            pred = model(dataset[idx][0].unsqueeze(0)).squeeze(0)
        if denorm:
            pred = denormalise(pred, MINMAX_PATH)

        df_NO2 = pd.concat([df_NO2, pd.DataFrame(pred[:, 0].detach().numpy())])
        df_O3 = pd.concat([df_O3, pd.DataFrame(pred[:, 1].detach().numpy())])
        df_PM10 = pd.concat([df_PM10, pd.DataFrame(pred[:, 2].detach().numpy())])
        df_PM25 = pd.concat([df_PM25, pd.DataFrame(pred[:, 3].detach().numpy())])

    df_NO2.rename(columns = {df_NO2.columns[0] : 'pred'}, inplace = True)
    df_O3.rename(columns = {df_O3.columns[0] : 'pred'}, inplace = True)
    df_PM10.rename(columns = {df_PM10.columns[0] : 'pred'}, inplace = True)
    df_PM25.rename(columns = {df_PM25.columns[0] : 'pred'}, inplace = True)

    print(df_NO2.shape)
    print(df_O3.shape)
    print(df_PM10.shape)
    print(df_PM25.shape)

    df_combined = pd.concat([
        df_NO2,
        df_O3,
        df_PM10,
        df_PM25,
    ])

    print(df_combined.shape)

    return df_combined


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


def plot_two_weeks_of_HGRU(
        model_HGRU: Any, test_dataset: torch.utils.data.Dataset, pair: int = 0
    ) -> None:
    """
    A snippet taken out of a big plotting notebook,
    plotting two weeks of HGRU predictions
    """
    pred_HGRU_NO2, gt_HGRU_NO2 = get_x_days_of_pred_and_gt(
        model_HGRU, test_dataset, pair, 'NO2', True, True)
    pred_HGRU_O3, gt_HGRU_O3 = get_x_days_of_pred_and_gt(
        model_HGRU, test_dataset, pair, 'O3', True, True)
    pred_HGRU_PM10, gt_HGRU_PM10 = get_x_days_of_pred_and_gt(
        model_HGRU, test_dataset, pair, 'PM10', True, True)
    pred_HGRU_PM25, gt_HGRU_PM25 = get_x_days_of_pred_and_gt(
        model_HGRU, test_dataset, pair, 'PM25', True, True)

    fig, axs = plt.subplots(4, 1, figsize = (10, 6))
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Computer Modern Serif",
            "axes.labelsize": 8,     # fontsize for x and y labels
            "xtick.labelsize": 8,    # fontsize of the tick labels
            "ytick.labelsize": 8,    # fontsize of the tick labels
            "legend.fontsize": 8,    # fontsize of the legen
        })
        
    sns.set_theme(style = 'ticks')
    sns.set_context("paper")
    sns.set_palette(sns.color_palette(["black", "#8B0000"]))

    axs[0].plot(gt_HGRU_NO2, color='black', linewidth=1.2, label=r'$\textrm{NO}_{2}$')
    axs[0].plot(pred_HGRU_NO2, color='#8B0000', linewidth=1.2)

    axs[1].plot(gt_HGRU_O3, color='black', linewidth=1.2, label=r'$\textrm{O}_{3}$')
    axs[1].plot(pred_HGRU_O3, color='#8B0000', linewidth=1.2)

    axs[2].plot(gt_HGRU_PM10, color='black', linewidth=1.2, label=r'$\textrm{PM}_{10}$')
    axs[2].plot(pred_HGRU_PM10, color='#8B0000', linewidth=1.2)

    axs[3].plot(gt_HGRU_PM25, color='black', linewidth=1.2, label=r'$\textrm{PM}_{2.5}$')
    axs[3].plot(pred_HGRU_PM25, color='#8B0000', linewidth=1.2)
    # x-tick moments every 24 hours
    x_date_moments = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336]
    print("Beware: the x-tick labels are hardcoded and can be changed in the function definition") 
    # Manually set the x-tick labels to be dates            
    x_date_labels = ['2021-12-09', '', '', '', '', '', '', '', '', '', '', '', '', '', '2021-12-23']   
    axs[0].set_xticks(x_date_moments)
    axs[1].set_xticks(x_date_moments)
    axs[2].set_xticks(x_date_moments)
    axs[3].set_xticks(x_date_moments)
    for ax in axs:
        ax.set(xlabel = '')
        ax.set(ylabel = '')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # For the top-three plots, remove the x-tick labels
    axs[0].set_xticklabels(['' for _ in axs[0].get_xticks()])
    axs[1].set_xticklabels(['' for _ in axs[1].get_xticks()])
    axs[2].set_xticklabels(['' for _ in axs[2].get_xticks()])
    # Manually adjust the legend position, as it is not automatically aligned
    axs[0].legend(loc = 'upper right', frameon = True, bbox_to_anchor = (1, 1))
    axs[1].legend(loc = 'upper right', frameon = True, bbox_to_anchor = (0.98596, 1))
    axs[2].legend(loc = 'upper right', frameon = True, bbox_to_anchor = (1.0084, 1))
    axs[3].legend(loc = 'upper right', frameon = True, bbox_to_anchor = (1.0135, 1))
    # Set x-tick labels to be dates, with TeX formatting
    axs[3].set_xticklabels([r'\textrm{' + label + '}' for label in x_date_labels])
    # Set y-ticks at 0, range/2, range
    axs[0].set_yticks([0, int(int(gt_HGRU_NO2.max()) / 2), int(gt_HGRU_NO2.max())])
    axs[1].set_yticks([0, int(int(gt_HGRU_O3.max()) / 2), int(gt_HGRU_O3.max())])
    axs[2].set_yticks([0, int(int(gt_HGRU_PM10.max()) / 2), int(gt_HGRU_PM10.max())])
    axs[3].set_yticks([0, int(int(gt_HGRU_PM25.max()) / 2), int(gt_HGRU_PM25.max())])
    # Line at y = 0
    axs[0].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    axs[1].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    axs[2].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    axs[3].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    # Save the figure as pdf (vector graphics)
    plt.savefig("plot_pollutants_HGRU_med.pdf",
                format = 'pdf',
                bbox_inches = 'tight',
                pad_inches = 0.015)

    plt.show()


def pretty_plot_loss(df: pd.DataFrame, model_name: str) -> None:
    """
    Does a loss plot, but a bit prettier

    :param df: DataFrame with the losses
    :param model_name: name of the model
    """
    plt.figure(figsize = (6, 4.5))
    sns.lineplot(df['L_train'], color = '#1C1678',
                 linewidth = 1.2, label = r'$L_{\textrm{train}}$')
    sns.lineplot(df['L_test'], color = '#FB6D48',
                 linewidth = 1.2, label = r'$L_{\textrm{validation}}$')
    plt.xlabel(r'$\textrm{Epoch} \longrightarrow$')
    plt.ylabel(r'$\textrm{Loss} \longrightarrow$')
    # A lot of {}'s to escape the TeX compiler
    plt.title(fr'$\textrm{{\textbf{{{model_name}}}}}$')

    plt.savefig(f"plot_loss_{model_name}.pdf",
             format = 'pdf',
             bbox_inches = 'tight',
             pad_inches = 0.015)

    plt.show()


def plot_modular_loss(df: pd.DataFrame, model_name: str) -> None:
    """
    Takes a dataframe with the losses of both the shared
    and branch part of the model, and plots them landscaped

    :param df: DataFrame with the losses, has to contain
               columns 'L_shared' and 'L_branch' (.csv's
               in results/final_losses happen to fit this
               description)
    :param model_name: name of the model
    """
    plt.figure(figsize = (12, 1.6))
    # Skip loss of first epoch as it skews the plot
    sns.lineplot(df['L_shared'][1:], color = '#1C1678',
                 linewidth = 1.2, label = r'$L_{\textrm{shared}}$')
    sns.lineplot(df['L_branch'][1:], color = '#8576FF',
                 linewidth = 1.2, label = r'$L_{\textrm{branches}}$')
    plt.xlabel(r'$\textrm{Epoch} \longrightarrow$')
    plt.ylabel(r'$\textrm{Loss} \longrightarrow$')
    # plt.title(fr'$\textrm{{\textbf{{{model_name}}}}}$')

    plt.savefig(f"plot_loss_dist_{model_name}.pdf",
                format = 'pdf',
                bbox_inches = 'tight',
                pad_inches = 0.015)

    plt.show()