from common.differentiable_function import DifferentiableFunction
import numpy as np

class WeightedQuadraticFunction(DifferentiableFunction):
    def __init__(self, center, weights):
        """
        f(x) = sum_i w_i * (x_i - c_i)^2

        Parameters:
            center (np.ndarray): Vector c specifying the center point.
            weights (np.ndarray): Vector w of positive weights (same size as center).
        """
        self.center = center
        self.weights = weights
        assert self.center.shape == self.weights.shape, "center and weights must match in shape"
        assert np.all(self.weights > 0), "weights must be positive"

    def forward(self, x):
        """
        Compute the weighted squared distance to center:
        f(x) = sum_i w_i * (x_i - c_i)^2

        Parameters:
            x (np.ndarray): Input vector of same shape as center.

        Returns:
            np.ndarray: f(x).
        """
        assert x.shape == self.center.shape
        self._input = x.copy()
        return np.sum(self.weights * (x - self.center)**2)

    def backward(self, grad_output):
        """
        Compute the gradient dL/dx using chain rule:
            dL/dx_i = dL/df * df/dx_i

        Parameters:
            grad_output (np.ndarray): Upstream gradient dL/df.

        Returns:
            np.ndarray: The gradient vector dL/dx.
        """
        return grad_output * 2 * self.weights * (self._input - self.center)
