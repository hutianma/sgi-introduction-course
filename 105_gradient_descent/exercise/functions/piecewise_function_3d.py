from common.differentiable_function import DifferentiableFunction
import numpy as np

class PiecewiseFunction3D(DifferentiableFunction):
    def __init__(self, center=1.0):
        """
        f(x) = softplus(x0) + (x1 - center)^2 + sin(x2)
        """
        self.center = center

    def forward(self, x):
        """
        Compute the value of the piecewise function:
        f(x) = softplus(x0) + (x1 - center)^2 + sin(x2)

        Parameters:
            x (np.ndarray): A 3D vector [x0, x1, x2].
        
        Returns:
            np.ndarray: The output f(x), shape (1,).
        """
        assert x.shape == (3,), "Input must be a 3D vector"
        self._input = x.copy()
        ...

    def backward(self, grad_output):
        """
        Compute the gradient of the loss with respect to the input x
        using the chain rule. Note: x0, x1, and x2 all require slightly
        different logic to compute the gradient so it is recommended to
        compute them separately.

        Parameters:
            grad_output (np.ndarray): The upstream gradient dL/df, i.e., the
                                      derivative of the loss L with respect to
                                      the output f(x).

        Returns:
            np.ndarray: The downstream gradient dL/dx, computed as:
                        dL/dx = dL/df * df/dx
                        The gradient should be of shape (3,).
        """
        x0, x1, x2 = self._input[0], self._input[1], self._input[2]
        ...
