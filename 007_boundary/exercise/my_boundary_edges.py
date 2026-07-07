import numpy as np
def my_boundary_edges(F):

    """Given a triangle mesh with face indices F, returns all unique oriented
    boundary edges as indices into the vertex array.
    Works only on manifold meshes.

    Parameters
    ----------
    F : (m,3) numpy int array.
        face index list of a triangle mesh

    Returns
    -------
    bE : (be,2) numpy int array.
        indices of boundary edges into the vertex array
    """
    bdr_edges = []
    for f in F:
        edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        for a,b in edges:
            edge = tuple(sorted((a,b)))
            if edge in bdr_edges:
                bdr_edges.remove(edge)
            else:
                bdr_edges.append(edge)

    return np.array(bdr_edges)

F = np.array([[0,3,1], [3,4,1], [1,4,2], [4,5,2], [3,6,4], [6,7,4], [4,7,5], [7,8,5]])
print(my_boundary_edges(F))

