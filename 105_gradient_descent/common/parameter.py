import numpy as np

class Parameter():
    """
    A class used to store the values and gradients for a parameter.

    Attributes
    ----------
    val : np.ndarray
        Array of learnable parameter values.
    grad: np.ndarray
        Array of associated gradient values.
    """
    def __init__(self, val):
        self.val = val
        self.grad = np.zeros_like(val)  # Initialize gradients to zero
