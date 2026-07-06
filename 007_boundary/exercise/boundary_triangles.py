import gpytoolbox as gpy, numpy as np

def boundary_triangles(F):
    """Return a list of boundary triangle indices for an input triangulation F.
    """

    # Compute boundary edges.
    bdry_edges = gpy.boundary_edges(F)

    # Find all triangles that contain both vertices of a boundary edge.
    # HINT: Look at the documentation of the `where` or `nonzero` function in
    # NumPy.
    bdry_tri_list = []
    for e in bdry_edges:
        i, j = e[0], e[1]
        contains_i = np.any(F == i, axis=1)
        contains_j = np.any(F == j, axis=1)
        mask = contains_i & contains_j          
        tris = np.nonzero(mask)[0]           
        bdry_tri_list.extend(tris.tolist())

    return np.array(bdry_tri_list)
