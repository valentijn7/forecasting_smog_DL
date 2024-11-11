# src/modelling/HGRU.py

# Class definition for the Hierarchical Gated Recurrent Unit (HGRU) model

from typing import List, Any
import torch
import torch.nn as nn

class Branch(nn.Module):
    """
    Assisting module for the HGRU branches: ignores all hidden 
    GRU states thereby allowing for the use of nn.Sequential.
    This gets pretty technical, but it's necessary to apply Dense
    (or, in other naming, Linear) layers to each time step of the
    output of the branches separately because the output of an RNN
    is a sequence of outputs, one for each time step. This does not
    just work with a single Dense layer, because it expects a 2D
    input and the output of PyTorch RNN is 3D (usually: (N_BATCHES,
    N_HOURS_Y, N_OUTPUT_UNITS)), in contrast to feedforward networks.
    By applying it 'loopwise' to each time step, this is solved.
    """

    def __init__(self, layers: List[Any], N_HOURS_Y):
        """
        Initializes the branch; needs the:
        - layers: the layers in the branch. These are of type
                  nn.GRU or nn.Linear (should be the last one)
        - N_HOURS_Y: the number of hours to sample from the output
        """
        super(Branch, self).__init__()
        # Adds the layers as a ModuleList, which later gets passed
        # and appended to the empty 'branches' ModuleList in the constructor
        self.layers = nn.ModuleList(layers)
        self.d_n_hours_y = N_HOURS_Y

    def forward(self, out):
        """
        Defines the forward pass of the module, taking the following steps:
        - Pass the input through all layers, except the last one
        - Apply the last layer to each time step (because the output of an RNN
          is a sequence of outputs, one for each time step), stack the outputs,
          and transpose them to (N_BATCHES, N_HOURS_Y, N_OUTPUT_UNITS)
        - Return the output
        """
        for layer in self.layers[:-1]:  # if an instance of nn.GRU, discard hidden states
                                        # (which should trigger every time, because [:-1])
            if isinstance(layer, nn.GRU):
                out, _ = layer(out)
                                        # apply dense layer to each time step, then stack,
                                        # which yields (N_HOURS_Y, N_BATCHES, N_OUTPUT_UNITS),
                                        # then transpose to (N_BATCHES, N_HOURS_Y, N_OUTPUT_UNITS)
        return torch.stack([self.layers[-1](out[:, idx, :])
                         for idx in range(-self.d_n_hours_y, 0)],
                        dim = 0).transpose(0, 1)

class HGRU(nn.Module):
    """
    Definition for a Hierarchical Gated Recurrent Unit (HGRU) model,
    or Multi-Branch GRU (MBGRU) model. In essence, it's a GRU model,
    but instead of a single connected recurrent bunch of neurons, its
    network flow gets (following one shared layer for shared representation)
    distributed over multiple branches, each with its own layer(s). As
    a result, the layers interfere less, hypothetically with overfitting,
    and a lot less parameters get used compared to a 'standard' GRU model.
    """

    def __init__(self,
                 N_HOURS_U: int = 72,      # 72 for the thesis
                 N_HOURS_Y: int = 24,      # 24 for the thesis
                 N_INPUT_UNITS: int = 10,  # 10 for the thesis
                 N_HIDDEN_LAYERS: int = 4, # 4 for the thesis       
                 N_HIDDEN_UNITS: int = 64, # 64 for the thesis       
                 N_BRANCHES = 4,           # 4 the thesis, and (always?) should be equal to:
                 N_OUTPUT_UNITS = 4        # this one
                ):
        """
        Constructor for the HGRU model. It takes in the parameters, assigns
        data members, loosely checks for divisibility, and initializes the
        input and shared layers, and the branches. The branches are initialized
        using the Branch module, defined above.
        """
        super(HGRU, self).__init__()
        self.d_n_hours_u = N_HOURS_U
        self.d_n_hours_y = N_HOURS_Y
        self.d_input_units = N_INPUT_UNITS
        self.d_hidden_layers = N_HIDDEN_LAYERS
        self.d_hidden_units = N_HIDDEN_UNITS
        self.d_branches = N_BRANCHES
        self.d_output_untis = N_OUTPUT_UNITS
        self._check_branch_parameters__()
        self.d_branch_units = self.d_output_untis // self.d_branches
        
        self.input_layer = nn.GRU(      # Initialize input layer and shared layer
            self.d_input_units, self.d_hidden_units, batch_first = True
            )
        self.shared_layer = nn.GRU(
            self.d_hidden_units, self.d_hidden_units, batch_first = True
            )
        self.branches = nn.ModuleList() # Initialize branches, using Branch module:
        for _ in range(self.d_branches):
            # First, initialize the (first and only) shared GRU layer
            branch_layers = [nn.GRU(self.d_hidden_units,
                                     self.d_hidden_units // self.d_branches,
                                     batch_first = True)]
            # Second, initialize the branch-specific layers
            branch_layers.extend([nn.GRU(self.d_hidden_units // self.d_branches,
                                          self.d_hidden_units // self.d_branches, 
                                          self.d_hidden_layers - 1,
                                          batch_first = True)])
            # Third, initialize the output layer
            branch_layers.append(nn.Linear(self.d_hidden_units // self.d_branches,
                                           self.d_output_untis // self.d_branches))
            # Lastly, pass the layers to the module, and append the branches
            self.branches.append(Branch(branch_layers, self.d_n_hours_y))

    def _check_branch_parameters__(self):
        """
        Checks if the neurons are evenly distributable
        over the branches, otherwise raises an error
        """
        if self.d_hidden_units % self.d_branches != 0:
            raise ValueError("N_HIDDEN_UNITS must be divisible by N_BRANCHES")
        if self.d_output_untis % self.d_branches != 0:
            raise ValueError("N_OUTPUT_UNITS must be divisible by N_BRANCHES")

    def forward(self, u):
        """
        Forward pass of the model:
        - Pass input through input layer, discard hidden states
        - Pass through shared layer, discard hidden states
        - Pass shared output through branches, in each only selecting the last N_HOURS_Y outputs
          (because an RNN works 'autoregressively', it produces just as many outputs as inputs,
          but we are only interested in the last N_HOURS_Y outputs)
        - Return the branches' outputs
        """
        u, _ = self.input_layer(u)      # pass through input layer, discard hidden states
                                        # pass through shared layer, discard hidden states
        shared_output, _ = self.shared_layer(u)
                                        # pass through branches, discard hidden states, in
                                        # seach electing the last N_HOURS_Y outputs
        return [branch(shared_output[:, -self.d_n_hours_y:, :]) for branch in self.branches]