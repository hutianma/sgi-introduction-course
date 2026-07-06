import numpy as np
import gpytoolbox as gpy

def write_tetrahedron():
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    F = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])

    gpy.write_mesh('004_reading_and_writing_a_mesh/data/tetrahedron.obj', V, F)

write_tetrahedron()