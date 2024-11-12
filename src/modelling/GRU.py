# src/modelling/HGRU.py

# Class definition for the Gated Recurrent Unit (GRU) model

import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    Defines a GRU network, with the member functions:
    - __init__: constructor
    - forward: defines the forward pass of the GRU
    """

    def __init__(self, # standard values are what was used in the thesis
                 N_HOURS_U: int = 72,
                 N_HOURS_Y: int = 24,
                 N_INPUT_UNITS: int = 10,
                 N_HIDDEN_LAYERS: int = 4,
                 N_HIDDEN_UNITS: int = 128,
                 N_OUTPUT_UNITS: int = 4):
        """
        Constructor; initializes the GRU model with the following parameters:

        :param N_HOURS_U: the number of hours in the input sequence
        :param N_HOURS_Y: the number of hours in the output sequence
        :param N_INPUT_UNITS: the number of input units
        :param N_HIDDEN_LAYERS: the number of hidden layers
        :param N_HIDDEN_UNITS: the number of hidden units
        :param N_BRANCHES: the number of branches
        :param N_OUTPUT_UNITS: the number of output units
        """ # First, iniatialize data members (denoted by d_ prefix)
        super(GRU, self).__init__()
        self.d_n_hours_u = N_HOURS_U
        self.d_n_hours_y = N_HOURS_Y
        self.d_input_units = N_INPUT_UNITS
        self.d_hidden_layers = N_HIDDEN_LAYERS
        self.d_hidden_units = N_HIDDEN_UNITS
        self.d_output_units = N_OUTPUT_UNITS
        
        # Second, init GRU layer(s) and a Dense output layer;
        # batch_first = 1 means that the input and output tensors
        # are provided as (batch, seq, feature), so batch dimension first
        self.gru = nn.GRU(self.d_input_units, self.d_hidden_units,
                          self.d_hidden_layers, batch_first = True)
        self.dense = nn.Linear(self.d_hidden_units, self.d_output_units)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU:
        - pass input tensor through big pile of GRU neurons, and
          leave only the last time step(s) of the output; and
        - apply dense layer to each time step of interest, then stack
          the results, and transpose the stack to the desired shape
        """
        out, _ = self.gru(u)            # take last time step(s) of each sequence in GRU output
        out = out[:, -self.d_n_hours_y:, :]
                                        # apply dense layer to each time step, then stack,
                                        # which yields (N_HOURS_Y, N_BATCHES, N_OUTPUT_UNITS),
                                        # then transpose to (N_BATCHES, N_HOURS_Y, N_OUTPUT_UNITS)
        return torch.stack([self.dense(out[:, idx, :])
                           for idx in range(-self.d_n_hours_y, 0)],
                           dim = 0).transpose(0, 1)