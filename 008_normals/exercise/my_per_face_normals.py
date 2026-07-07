import gpytoolbox as gpy, numpy as np
import polyscope as ps
def my_per_face_normals(V,F, unit_norm=True):
  """Vector perpedicular to all faces on a mesh

    Computes per face (optionally unit) normal vectors for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,d) numpy int array
        face index list of a triangle mesh
    unit_norm : bool, optional (default True)
        Whether to normalize each face's normal before outputting

    Returns
    -------
    N : (n,d) numpy double array
        Matrix of per-face normals
  """
  v0 = V[F[:,0]]
  v1 = V[F[:,1]]
  v2 = V[F[:,2]]

  e1 = v1 - v0
  e2 = v2 - v0
  N = np.cross(e1, e2)

  if unit_norm:
    N /= np.linalg.norm(N, axis=1, keepdims=True)

  return N

V,F = gpy.read_mesh('/Users/huyufan/Documents/GitHub/sgi-introduction-course/008_normals/data/spot.obj')
N = my_per_face_normals(V,F)
ps.init()
ps_spot = ps.register_surface_mesh("spot", V, F, smooth_shade=True)
ps_spot.add_vector_quantity("per-face normals", N, defined_on='faces', enabled=True)
ps.show()
