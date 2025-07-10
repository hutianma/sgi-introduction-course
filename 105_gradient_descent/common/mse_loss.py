from common.differentiable_function import DifferentiableFunction
import numpy as np

class MSELoss(DifferentiableFunction):
    """
    Mean Squared Error (MSE) loss function.

    Computes:
        L = (1/N) * sum((y_pred - y_target)^2)

    Methods
    -------
    forward(y_pred, y_target)
        Computes the MSE loss.

    backward(y_pred, y_target)
        Computes the gradient of the loss with respect to y_pred.
    """

    def forward(self, y_pred, y_target):
        """
        Compute the MSE loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.
        y_target : np.ndarray
            Target values.

        Returns
        -------
        float
            Scalar MSE loss.
        """
        self.diff = y_pred - y_target
        self.n = y_pred.size
        return np.mean(self.diff ** 2)

    def backward(self):
        """
        Compute the gradient of the MSE loss w.r.t. y_pred.

        Returns
        -------
        np.ndarray
            Gradient w.r.t. y_pred.
        """
        return (2.0 / self.n) * self.diff