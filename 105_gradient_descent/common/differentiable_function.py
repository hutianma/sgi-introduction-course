from abc import ABC, abstractmethod

class DifferentiableFunction(ABC):
    """
    Abstract base class for differentiable functions.
    """

    @abstractmethod
    def forward(self, x):
        """
        Compute the function output f(x) for an input x. Also saves the input x to
        self._input for use in the backward pass.
        
        Parameters:
            x (np.ndarray): Input value.
        
        Returns:
            np.ndarray: The result of f(x).
        """
        pass

    @abstractmethod
    def backward(self, grad_output):
        """
        Given the upstream gradient (dL/df), compute and return the downstream
        gradient (dL/dx) using the chain rule.

        Parameters:
            grad_output (np.ndarray): The upstream gradient dL/df, where L is some scalar
                                 loss depending on the output f(x).
        
        Requires:
            self._input (np.ndarray): The input value (x) at which the forward pass was
                                 evaluated.

        Returns:
            np.ndarray: The downstream gradient dL/dx.
        """
        pass
