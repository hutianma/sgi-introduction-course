class Optimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.learning_rate = learning_rate
    
    def step(self):
        """
        Perform a single optimization step.
        
        This method updates the parameters using the gradients stored in each parameter.
        The update rule for a parameter is:
            value = value - (learning_rate * gradient)
        """
        ...

    def zero_grad(self):
        """
        Reset the gradients to zero.
        """
        self.params.grad = 0.0
