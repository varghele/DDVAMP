import torch
import torch.nn as nn
from typing import List, Union, Optional

class LinearLayer:
    """
    Factory class for creating linear layers with customizable initialization and components.
    """

    @staticmethod
    def create(
            d_in: int,
            d_out: int,
            bias: bool = True,
            activation: Optional[nn.Module] = None,
            dropout: float = 0,
            weight_init: Union[str, float, int, None] = 'xavier',
            gain: float = 1.0
    ) -> List[nn.Module]:
        """
        Creates a linear layer with optional activation and dropout.

        Parameters
        ----------
        d_in : int
            Input dimension
        d_out : int
            Output dimension
        bias : bool, optional
            Whether to include bias, by default True
        activation : nn.Module, optional
            Activation function to use, by default None
        dropout : float, optional
            Dropout probability, by default 0
        weight_init : Union[str, float, int, None], optional
            Weight initialization method ('xavier', 'identity', 'kaiming')
            or constant value, by default 'xavier'
        gain : float, optional
            Gain factor for xavier initialization, by default 1.0

        Returns
        -------
        List[nn.Module]
            List of layer components

        Raises
        ------
        TypeError
            If activation is not a valid nn.Module
        ValueError
            If weight_init method is not recognized
        """
        # Create linear layer
        linear = nn.Linear(d_in, d_out, bias=bias)

        # Initialize weights
        with torch.no_grad():
            if weight_init == 'xavier':
                nn.init.xavier_uniform_(linear.weight, gain=gain)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init == 'kaiming':
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init == 'identity':
                nn.init.eye_(linear.weight)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif isinstance(weight_init, (int, float)):
                nn.init.constant_(linear.weight, weight_init)
                if bias:
                    nn.init.zeros_(linear.bias)
            elif weight_init is not None:
                raise ValueError(f"Unsupported weight initialization: {weight_init}")

        # Build layer sequence
        layers = [linear]

        # Add activation if specified
        if activation is not None:
            if not isinstance(activation, nn.Module):
                raise TypeError(f'Activation {activation} is not a valid torch.nn.Module')
            layers.append(activation)

        # Add dropout if specified
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        return layers

    @staticmethod
    def get_activation(activation_type: str) -> nn.Module:
        """
        Get activation function by name.

        Parameters
        ----------
        activation_type : str
            Name of activation function ('relu', 'tanh', 'sigmoid', etc.)

        Returns
        -------
        nn.Module
            Activation function
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }
        if activation_type not in activations:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        return activations[activation_type]
