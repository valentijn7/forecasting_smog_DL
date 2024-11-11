# src/modelling/EarlyStopper.py

import numpy as np
import copy as cp

class EarlyStopper:
    """
    Early stops training if validation loss (L_val)
    doesn't improve after a given patience
    """
    def __init__(self, patience = 5, verbose = False):
        """
        Initialize early stopping object, including start params
        
        :param patience: number of epochs to wait before stopping
        :param verbose: print information about early stopping
        """
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_val_loss = np.Inf
        self._early_stop = False
        self._best_model = None

    # @porperty is a decorator that makes a method behave like a read-only attribute
    @property
    def best_model(self):
        """
        Accessor to to the best saved model, which is useful after early stopping
        and a previously best model has to be restored and retrieved
        """
        return self._best_model

    def __call__(self, val_loss, epoch, model):
        """
        Call function for early stopping, which:
        - checks if validation loss has improved (with a small margin
          to avoid stopping due to numerical errors or marginal improvements)
        - else:
            - if the counter is zero, save the best model (with a deep copy, so
                that the model is not the standard reference-based copying)
            - increase the counter
            - if the counter exceeds the patience, early stop
        """
        if val_loss < self._best_val_loss - 0.00001:
            self._best_val_loss = val_loss
            self._counter = 0
        else:
            if self._counter == 0:
                self._best_model = cp.deepcopy(model)
            self._counter += 1
            if self._counter >= self._patience:
                self._early_stop = True
                if self._verbose:
                    print(f"EarlyStopper: stopping at epoch {epoch}", end = ' ')
                    print(f"with best_val_loss = {self._best_val_loss:.6f}\n")
                    return True
        return False