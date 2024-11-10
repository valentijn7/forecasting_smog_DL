import numpy as np
import copy as cp

class EarlyStopper:
    """Early stops training if L_val doesn't improve after a given patience"""

    def __init__(self, patience = 5, verbose = False):
        """Initialize early stopping object"""
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_val_loss = np.Inf
        self._early_stop = False
        self._best_model = None

    @property
    def best_model(self):
        """Accessor to to the best saved model"""
        return self._best_model

    def __call__(self, val_loss, epoch, model):
        """Call function for early stopping"""
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