import numpy as np
import polyscope as ps
import gpytoolbox as gpy
def tangents(V,F):
    """
    Computes two orthogonal, oriented tangent vectors for each face in a
    triangle mesh.
    """

    # Extract the first edge of each face and normalize it.

    # Extract the second edges and project onto the orthogonal complement of E1.

    # Normalize to get unit vectors
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e1 = v1 - v0
    T1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)

    e2 = v2 - v0
    proj = np.sum(e2 * T1, axis=1, keepdims=True)
    T2 = e2 - proj * T1
    T2 = T2 / np.linalg.norm(T2, axis=1, keepdims=True)

    return T1, T2

V, F = gpy.read_mesh('/Users/huyufan/Documents/GitHub/sgi-introduction-course/008_normals/data/spot.obj')
T1, T2 = tangents(V, F)

ps.init()
ps_spot = ps.register_surface_mesh("spot", V, F, smooth_shade=True)
ps_spot.add_vector_quantity("tangent T1", T1, defined_on='faces', enabled=True, color=(1, 0, 0))
ps_spot.add_vector_quantity("tangent T2", T2, defined_on='faces', enabled=True, color=(0, 1, 0))
ps.show()