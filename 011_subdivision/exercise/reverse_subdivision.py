import numpy as np
import gpytoolbox as gpy
import scipy as sp

def reverse_subdivision(V, F, uu, k):
    """Given a function uu on a mesh which has been subdivided k times from the
    coarse mesh V,F, reconstruct a function u on the coarse mesh V,F.
    """

    Vu, Fu, S = gpy.subdivide(
        V, F,
        method="loop",
        iters=k,
        return_matrix=True
    )

    u = sp.sparse.linalg.lsqr(S, uu)[0]

    return u

