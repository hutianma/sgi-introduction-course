# Write your own gradient descent!

In this module, we will write our own simple framework for performing optimization using gradient descent. This framework will mimic a lot of the structure of how PyTorch works internally!

Before starting, it is recommended that you look through the code in `common/` as this will be helpful for understanding the structure of the exercises going forward. For example, the structure of `common/mse_loss.py` can serve as an example with a similar format to the functions that you will write in this module.

# Exercises

## Manually Implementing Gradient Computations

### Task 1
Implement the `forward()` and `backward()` methods for the 1D quadratic function class in `exercise/functions/quadratic_function_1d.py`. Note: per PyTorch convention the forward pass stores the input in `self._input` to be used for the gradient computation in the backward pass. If successfully completed, you should be able to pass the first test case (1/4) when running `main.py`.

### Task 2
Implement the `forward()` and `backward()` methods for the weighted quadratic function class in `exercise/functions/weighted_quadric_function.py`. Note, this function has multiple input variables that each require their own gradient! Since both the input variables and gradient variables are stored in NumPy arrays, we can just put the gradient for the ith variable in the ith index of the gradient array that we return. If successfully completed, you should be able to pass the second test case (2/4) when running `main.py`.

### Task 3
Implement the `forward()` and `backward()` methods for the 3D piecewise function class in `exercise/functions/piecewise_function_3d.py`. This function is also a function of multiple variables (3). However, unlike the previous task, each gradient for each variable needs to be computed with slightly different logic, so it is recommended that you split these up. We can still store the gradients for the ith variable in the ith index of the returned gradient array. If successfully completed, you should be able to pass the third test case (3/4) when running `main.py`.

## Filling in the Training Loop

### Task 4
Complete the update step `step()` method in `exercise/optimizer.py`. Recall our update rule is $x_{n+1} = x_{n} - \eta \frac{dL}{dx}$ where $\eta$ is our learning rate.

### Task 5
Fill in the forward pass, loss computation, and backward pass in the `train()` method in the `Trainer` class in `exercise/trainer.py`. For the forward pass, take the input parameter(s) and pass them through the `forward()` method for each of the functions in order to obtain some final forward pass output. Then, to compute the loss, use the loss function on the target and the forward pass output. To compute the backward pass, run the `backward()` method of the loss class and then take the output gradient and use it to chain through all of the `backward()` methods of all of the functions now in *REVERSE* order.

If both tasks 4 and 5 have been successfully completed, you should be able to pass all four test cases when running `main.py`!

## Debugging

To help with debugging, you may alter the code in `main.py` to enable "verbose" mode when running the optimizations. This will print out information about the loss at each iteration as well as the final parameter value, the final loss, the target value, and the final forward pass value. Feel free to also add print statements to your own code to help debug.

Good luck, and make sure to post in the Slack with any questions!
