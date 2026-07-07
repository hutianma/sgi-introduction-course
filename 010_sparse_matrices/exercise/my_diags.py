import numpy as np, scipy as sp
def my_diags(v, diagonals, shape):
    """This functin constructs a diagonal or banded sparse matrix, just
    like SciPy's sparse.diag
    """
    m, n = shape

    i_list = []
    j_list = []
    k_list = []

    for k, d in zip(v, diagonals):
        if d >= 0:
            length = min (m, n-d, len(v))
            i_list.append(np.arange(length))
            j_list.append(np.arange(d, d+length))
            k_list.append(v[:length])
        
        else:
            length = min (m, n+d, len(v))
            i_list.append(np.arange(length))
            j_list.append(np.arange(d, d+length))
            k_list.append(v[:length])

    i = np.concatenate(i_list)
    j = np.concatenate(j_list)
    k = np.concatenate(k_list)

    return sp.sparse.coo_matrix((k, (i, j)), shape=shape)
