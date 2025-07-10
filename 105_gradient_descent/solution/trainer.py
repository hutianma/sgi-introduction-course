from common.parameter import Parameter
from solution.optimizer import Optimizer

class Trainer:
    def __init__(self, functions, loss_fn, x, lr, log_freq=100, verbose=False):
        """
        Initialize the trainer with a list of functions, a loss function, and an initial parameter value
        
        Parameters::
            functions (list[DifferentiableFunction]): List of differentiable functions to apply in sequence.
            loss_fn (DifferentiableFunction): The loss function to compute the error.
            x (float): Initial value for the parameter to be optimized.
            lr (float): Learning rate for the optimizer.
        """
        self.functions = functions
        self.loss_fn = loss_fn
        self.optimizer = Optimizer(Parameter(x), learning_rate=lr)
        self.log_freq = log_freq
        self.verbose = verbose

    def train(self, target, epochs=1000):
        """
        Train the model by applying the functions in sequence,
        computing the loss, and updating parameters.

        Parameters::
            target (np.ndarray): The target value(s)
            epochs (int): Number of training epochs
        
        Returns:
            np.ndarray: The final parameter value(s) after training.
        """
        for epoch in range(epochs):
            # Forward pass
            x = self.optimizer.params.val
            for fn in self.functions:
                x = fn.forward(x)
            
            # Compute loss
            loss = self.loss_fn.forward(x, target)
            
            # Backward pass
            grad = self.loss_fn.backward()
            for fn in reversed(self.functions):
                grad = fn.backward(grad)

            # Assign gradients to parameters
            self.optimizer.params.grad = grad

            # Parameter update
            self.optimizer.step()

            # Optional: log progress
            if (epoch % self.log_freq == 0 or epoch == epochs - 1) and self.verbose:
                print(f"Epoch {epoch}: loss = {loss:.6f}")

        if self.verbose:
            print("Training complete.")
            print(f"Final parameter value(s): {self.optimizer.params.val}")
            print(f"Final loss: {loss:.6f}")
            print(f"Target value(s): {target}, Final value(s): {x}")
        return self.optimizer.params.val