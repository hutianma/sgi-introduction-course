from common.differentiable_function import DifferentiableFunction

class QuadraticFunction1D(DifferentiableFunction):
    def __init__(self, a=1.0, b=0.0, c=0.0):
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        """
        Compute the value of the quadratic function f(x) = ax^2 + bx + c.

        Parameters:
            x (np.ndarray): Input value(s).

        Returns:
            np.ndarray: The result of f(x).
        """
        assert x.shape == (1,)
        self._input = x.copy()  # Save input for backward pass
        return self.a * x**2 + self.b * x + self.c

    def backward(self, grad_output):
        """
        Compute the gradient of the loss with respect to the input x
        using the chain rule.

        Parameters:
            grad_output (np.ndarray): The upstream gradient dL/df, i.e., the derivative
                                of the loss L with respect to the output f(x).
        
        Requires:
            self._input (np.ndarray): The input values (x) at which the forward pass was evaluated.

        Returns:
            np.ndarray: The downstream gradient dL/dx, computed as:
                        dL/dx = dL/df * df/dx
        """
        x = self._input # Retrieve the saved input value
        df_dx = 2 * self.a * x + self.b
        return grad_output * df_dx
