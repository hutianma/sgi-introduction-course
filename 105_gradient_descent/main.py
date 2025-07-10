from common.mse_loss import MSELoss
from exercise.functions.quadratic_function_1d import QuadraticFunction1D
from exercise.functions.weighted_quadric_function import WeightedQuadraticFunction
from exercise.functions.piecewise_function_3d import PiecewiseFunction3D
from exercise.trainer import Trainer
from solution.trainer import Trainer as SolTrainer

import numpy as np

def test_quadratic_function_1d():
    """
    Test the 1D quadratic function optimization using gradient descent.
    """
    functions = [QuadraticFunction1D(a=1.0, b=-6.0, c=9.0)]
    trainer = SolTrainer(
        functions=functions,
        loss_fn=MSELoss(),
        x=np.random.uniform(-10.0, 10.0, size=(1,)),
        lr=1e-3,
        log_freq=500
    )
    val = trainer.train(target=np.array([9.0]), epochs=1000)
    assert np.isclose(val, np.array([0.0]), atol=1e-2) or \
        np.isclose(val, np.array([6.0]), atol=1e-2), \
        f"Expected final value close to 0.0 or 6.0, got {val}"

def test_weighted_quadratic_function_2d():
    """
    Test the weighted quadratic function optimization in 2D using gradient descent.
    """
    functions = [WeightedQuadraticFunction(center=np.array([0.0, 0.0]),
                        weights=np.array([1.0, 2.0]))]
    trainer = SolTrainer(
        functions=functions,
        loss_fn=MSELoss(),
        x=np.array([1.0, 2.0]),
        lr=1e-3,
        log_freq=100
    )
    val = trainer.train(target=np.array([1.0]), epochs=600)
    assert np.allclose(val, np.array([0.5460896, 0.59245319]), atol=1e-2), \
        f"Expected final value close to [0.5460896, 0.59245319], got {val}"

def test_piecewise_function_3d():
    """
    Test the piecewise function optimization in 3D using gradient descent.
    """
    functions = [PiecewiseFunction3D()]
    trainer = SolTrainer(
        functions=functions,
        loss_fn=MSELoss(),
        x=np.array([1.0, 2.0, 3.0]),
        lr=1e-3,
        log_freq=500
    )
    val = trainer.train(target=np.array([1.0]), epochs=1800)
    assert np.allclose(val, np.array([0.69887216, 1.42242816, 3.42595028]), atol=1e-2), \
        f"Expected final value close to [0.69887216, 1.42242816, 3.42595028], got {val}"

def test_multiple_functions():
    """
    Test the optimization of multiple functions composed in sequence using gradient descent.
    """
    functions = [
        PiecewiseFunction3D(),
        QuadraticFunction1D(a=3.0, b=-4.0, c=12.0),
    ]
    trainer = Trainer(
        functions=functions,
        loss_fn=MSELoss(),
        x=np.array([1.0, 2.0, 0.0]),
        lr=1e-4,
        log_freq=100,
    )
    val = trainer.train(target=np.array([73.0]), epochs=1000)
    assert np.allclose(val, np.array([1.43714743, 2.73039699, 0.61913697]), atol=1e-2), \
        f"Expected final value close to [1.43714743, 2.73039699, 0.61913697], got {val}"

def main():
    """
    Main function to run all tests.
    """
    test_quadratic_function_1d()
    print("1/4 -- 1D Quadratic Function Test Passed!")
    test_weighted_quadratic_function_2d()
    print("2/4 -- 2D Weighted Quadratic Function Test Passed!")
    test_piecewise_function_3d()
    print("3/4 -- 3D Piecewise Function Test Passed!")
    test_multiple_functions()
    print("4/4 -- Multiple Composed Functions Test Passed!")

if __name__ == '__main__':
    main()