# src/modelling/train.py

# Scripts that train fully-connected and hierarchical
# recurrent neural networks, indicated by either
# nothing, or 'hierarchical' in the name, respectively
# #! Add the GRU training-specific functions as well!

from typing import Any, List, Dict, Tuple
import numpy as np
import pandas as pd
import torch


def init_model(hp: Dict[str, Any]) -> Any:
    """
    Initializes standard model
    
    :param hp: dictionary of hyperparameters
    :return: the initialized model
    """
    return hp['model_class'](int(hp['input_units']),
                             int(hp['hidden_layers']), 
                             int(hp['hidden_units']),
                             int(hp['output_units']))


def init_mb_model(
        hp: Dict[str, Any],
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> Any:
    """
    Inits multi-branched, or hierarchical, model,
    by calling its constructor with the hyperparameters

    :param hp: dictionary of hyperparameters
    :return: the initialized model
    """
    return hp['model_class'](hp['n_hours_u'],
                             hp['n_hours_y'],
                             hp['input_units'],
                             hp['hidden_layers'],
                             hp['hidden_units'],
                             hp['branches'],
                             hp['output_units']
                             ).to(device)


def init_main_optimizer(model: Any, hp: Dict[str, Any]) -> Any:
    """
    Initializes the optimizer for the shared layer,
    aka the main optimizer

    :param model: the model to optimize
    :param hp: dictionary of hyperparameters
    :return: the initialized optimizer
    """
    return hp['Optimizer'](model.parameters(),
                           lr = hp['lr_shared'], 
                           weight_decay = hp['w_decay'])


def init_branch_optimizers(model: Any, hp: Dict[str, Any]) -> Any:
    """
    Initializes the optimizers for each branch
    
    :param model: the model to optimize
    :param hp: dictionary of hyperparameters
    :return: list of initialized optimizers
    """
    optimizers = []
    for branch in model.branches:
        optimizers.append(hp['Optimizer'](branch.parameters(), 
                                          lr = hp['lr_branch'], 
                                          weight_decay = hp['w_decay']))
    return optimizers


def init_main_scheduler(optimizer: Any, hp: Dict[str, Any]) -> Any:
    """
    Initializes the scheduler for the shared layer

    :param optimizer: the optimizer to schedule
    :param hp: dictionary of hyperparameters
    :return: the initialized scheduler
    """
    return hp['scheduler'](optimizer, **hp['scheduler_kwargs'])


def init_branch_schedulers(optimizers: List[Any], hp: Dict[str, Any]) -> List[Any]:
    """
    Initializes the schedulers for each branch
    
    :param optimizers: list of optimizers to schedule
    :param hp: dictionary of hyperparameters
    :return: list of initialized schedulers
    """
    schedulers = []
    for optimizer in optimizers:
        schedulers.append(hp['scheduler'](optimizer, **hp['scheduler_kwargs']))
    return schedulers


def init_early_stopper(hp: Dict[str, Any], verbose: bool = False) -> Any:
    """
    Initializes early stopping object
     
    :param hp: dictionary of hyperparameters
    :param verbose: whether EarlyStopper should be verbose
    :return: the initialized EarlyStopper
    """
    return hp['early_stopper'](hp['patience'], verbose)


def schedulers_epoch(
        main_scheduler: Any, secondary_schedulers: List[Any], val_loss: bool = None
    ) -> None:
    """
    Performs a scheduler step for each scheduler

    :param main_scheduler: the main scheduler
    :param secondary_schedulers: list of branch schedulers
    :param val_loss: the validation loss
    """
    main_scheduler.step(val_loss)
    for scheduler in secondary_schedulers:
        scheduler.step(val_loss)


def print_epoch_loss(epoch: int, train_losses: List[float],
                     val_losses: List[float], x: int = 5) -> None:
    """
    Prints the train and validation losses per x epochs
    
    :param epoch: the current epoch
    :param train_losses: list of training losses
    :param val_losses: list of validation losses
    :param x: print every x epochs
    """
    if ((epoch + 1) % x == 0) or (epoch == 0):
        print("Epoch: {} \tLtrain: {:.6f} \tLval: {:.6f}".format(
              epoch + 1, train_losses[epoch], val_losses[epoch]))


def training_epoch_shared_layer(
        model: Any, optimizer: Any, loss_fn: Any, 
        train_loader: torch.utils.data.DataLoader,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> float:
    """
    Trains the shared layer of the model for one epoch by:
    - freezing the branches
    - looping over the training set
    - doing the forward pass and calculating the loss
    - doing the backward pass and updating the weights
    - returning the training loss
    
    :param model: the model to train
    :param optimizer: the optimizer to use
    :param loss_fn: the loss function
    :param train_loader: the training data loader
    :param device: the device to train on
    :return: the training loss
    """                                 # freeze the branches; they do not
                                        # need to be trained in this step
    for param in model.branches.parameters():
        param.requires_grad_(False)

    train_loss = np.float64(0)          # loop over all batches in the training set
    for batch_train_u, batch_train_y in train_loader:
            batch_train_u = batch_train_u.to(device)
            batch_train_y = batch_train_y.to(device)
                                        # do forward pass, calculate loss
            batch_preds = torch.cat(model(batch_train_u), dim = 2)
            batch_loss = loss_fn(batch_preds, batch_train_y)
            train_loss += batch_loss.item()

            optimizer.zero_grad()       # do backward pass, optimize weights
            batch_loss.backward()
            optimizer.step()
    return train_loss


def training_epoch_branches(
        model: Any, optimizers: Any, loss_fn: Any,
        train_loader: torch.utils.data.DataLoader,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> float:
    """
    Trains the branches of model for one by doing the
    complementary to what training_epoch_shared_layer() does:
    - unfreezing the branches
    - freezing the shared layer
    - looping over the training set
    - for each branch:
        - doing the forward pass and calculating the loss
        - doing the backward pass and updating the weights
    - unfreezing the shared layer
    - returning the training loss
    
    :param model: the model to train
    :param optimizers: the optimizers to use
    :param loss_fn: the loss function
    :param train_loader: the training data loader
    :param device: the device to train on
    :return: the training loss
    """                                 # unfreeze the branches,
                                        # freeze the shared layer
    for param in model.branches.parameters():
        param.requires_grad_(True)
    for param in model.shared_layer.parameters():
        param.requires_grad_(False)
    
    train_loss = np.float64(0)
    model.train()                       # loop over all batches
    for batch_train_u, batch_train_y in train_loader:
        batch_train_u = batch_train_u.to(device)
        batch_train_y = batch_train_y.to(device)
                                        # for every batch, loop over the branches
        for idx, optimizer in enumerate(optimizers):
                                        # do forward pass, calculate loss
            branch_batch_loss = loss_fn(
                torch.cat(model(batch_train_u), dim = 2)[:, :, idx],
                batch_train_y[:, :, idx] 
            )
            train_loss += branch_batch_loss.item()
                                        
            optimizer.zero_grad()       # do backward pass, optimize weights
            branch_batch_loss.backward()
            optimizer.step()
                                        # finally, unfreeze the shared layer
                                        # and return the training loss
    for param in model.shared_layer.parameters():
            param.requires_grad_(True)
    return train_loss


def validation_epoch(
        model: Any, loss_fn: Any, val_loader: torch.utils.data.DataLoader,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> float:
    """
    Evaluates the model on the validation set for one epoch

    :param model: the model to evaluate
    :param loss_fn: the loss function
    :param val_loader: the validation data loader
    :return: the validation loss
    """
    val_loss = np.float64(0)            # set loss to 0 and model to eval mode
    model.eval()                        
    with torch.no_grad():               # don't calculate gradients
                                        # loop over all batches
        for batch_val_u, batch_val_y in val_loader:
                                        # move data to device
            batch_val_u = batch_val_u.to(device)
            batch_val_y = batch_val_y.to(device)
                                        # do forward pass, concatenate
                                        # predictions, calculate loss
            batch_preds = torch.cat(model(batch_val_u), dim = 2)
            val_loss += loss_fn(batch_preds, batch_val_y).item()
    return val_loss


def train_hierarchical(
        hp: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader, verbose: bool = False
    ) -> Tuple[Any, List[float], List[float], List[float], List[float]]:
    """
    Main training function for training a hierarchical model, with the 
    help of several helper functions. The training process is as follows:
    - init model, optimizers, schedulers, early stopper, etc.
    - for each epoch:
        - train the shared layer
        - train the branches
        - evaluate on the validation set
        - perform a scheduler step
        - print the average train and validation losses per epoch
        - check if the early stopper should stop the training
    - return the best model, and the train and validation losses
    """
    model = init_mb_model(hp)
    main_optimizer = init_main_optimizer(model, hp)
    branch_optimizers = init_branch_optimizers(model, hp)
    shared_scheduler = init_main_scheduler(main_optimizer, hp)
    branch_schedulers = init_branch_schedulers(branch_optimizers, hp)
    early_stopper = init_early_stopper(hp, verbose)
    loss_fn = hp['loss_fn']
    shared_losses, branch_losses = [], []
    train_losses, val_losses = [], []

    for epoch in range(hp['epochs']):
                                        # alternately train shared
                                        # and branched model part 
        shared_losses.append(training_epoch_shared_layer(
            model, main_optimizer, loss_fn, train_loader
            ))
        branch_losses.append(training_epoch_branches(
            model, branch_optimizers, loss_fn, train_loader
            ))
                                        # calculate training and validation loss
        train_loss = validation_epoch(model, loss_fn, train_loader)
        val_loss = validation_epoch(model, loss_fn, val_loader)
        schedulers_epoch(shared_scheduler, branch_schedulers, val_loss)
                                        # save/print losses per batch for each epoch
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print_epoch_loss(epoch, train_losses, val_losses, 1) if verbose else None
                
        if early_stopper(val_losses[epoch], epoch, model):
            break
    return early_stopper.best_model, train_losses, val_losses, shared_losses, branch_losses


def init_model(hp):
    """Initializes the model"""
    return hp['model_class'](int(hp['input_units']), int(hp['hidden_layers']), 
                             int(hp['hidden_units']), int(hp['output_units']))


def print_epoch_loss(epoch, train_losses, val_losses, x = 5):
    """Prints the train and validation losses per x epochs"""
    if ((epoch + 1) % x == 0) or (epoch == 0):
        print("Epoch: {} \tLtrain: {:.6f} \tLval: {:.6f}".format(
              epoch + 1, train_losses[epoch], val_losses[epoch]))


def train(
        hp: Dict[str, Any], train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader, verbose: bool = True,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> Tuple[Any, List[float], List[float]]:
    """
    Trains a fully connected model through a sequence of steps:
    - initialize the model, optimizer, and loss function
    - for each epoch (until early stopping is triggered):
        - for each batch in the training set
            - do a forward pass
            - calculate the loss
            - do a backward pass
            - update the weights
        - for each batch in the validation set
            - calculate the validation loss
        - save the average losses per batch for each epoch
        - do a scheduler step
    - return the trained model and the train and validation losses

    :param hp: dictionary of hyperparameters
    :param train_loader: DataLoader to get batches from
    :param val_loader: DataLoader to get batches from
    :param verbose: whether to print the losses per epoch
    :param device: the device to train on
    :return: the trained model, and the train and validation
    """
    model = init_model(hp).to(device)
    optimizer = init_main_optimizer(model, hp)
    lr_scheduler = init_main_scheduler(optimizer, hp)
    early_stopper = init_early_stopper(hp, verbose)
    loss_fn = hp['loss_fn']
    train_losses, val_losses = [], []

    for epoch in range(hp['epochs']):   # for each epoch:
        train_loss = 0.0                # reset the losses
        val_loss = 0.0
        model.train()                   

        for batch_train_u, batch_train_y in train_loader:
                                        # move data to GPU if available
            batch_train_u = batch_train_u.to(device)
            batch_train_y = batch_train_y.to(device)
                                        # forward pass
            batch_loss = loss_fn(model(batch_train_u),
                                 batch_train_y.squeeze(1))
            train_loss += batch_loss.item()

            optimizer.zero_grad()       # backward pass
            batch_loss.backward()
            optimizer.step()            # update the weights

        model.eval()                    # evaluate the model
        with torch.no_grad():
            for batch_val_u, batch_val_y in val_loader:
                                        # move data to GPU if available
                batch_val_u = batch_val_u.to(device)
                batch_val_y = batch_val_y.to(device)
                                        # calculate validation loss
                val_loss += loss_fn(model(batch_val_u), batch_val_y).item()
                                        # save average losses per batch for each epoch
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        lr_scheduler.step(val_loss)     
        print_epoch_loss(epoch, train_losses, val_losses) if verbose else None

        if early_stopper(val_losses[epoch], epoch, model):
            break
    return early_stopper.best_model, train_losses, val_losses