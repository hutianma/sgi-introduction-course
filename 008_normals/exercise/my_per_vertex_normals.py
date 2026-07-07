import gpytoolbox as gpy, numpy as np
import polyscope as ps

def my_per_vertex_normals(V,F):
  """Normal vectors to all vertices on a mesh

    Computes area-weighted per-vertex unit normal vectors for a triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,d) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    N : (m,d) numpy double array
        Matrix of per-vertex normals
  """
  N = np.zeros(V.shape) 
  for f in F:
    v0 = V[f[0]]
    v1 = V[f[1]]
    v2 = V[f[2]]

    e1 = v1 - v0
    e2 = v2 - v0
    face_n = np.cross(e1, e2)

    N[f[0]] += face_n
    N[f[1]] += face_n
    N[f[2]] += face_n

  return N / np.linalg.norm(N, axis=1, keepdims=True)
  

V,F = gpy.read_mesh('/Users/huyufan/Documents/GitHub/sgi-introduction-course/008_normals/data/spot.obj')
N = my_per_vertex_normals(V,F)
ps.init()
ps_spot = ps.register_surface_mesh("spot", V, F, smooth_shade=True)
ps_spot.add_vector_quantity("per-vertex normals", N, defined_on='vertices', enabled=True)
ps.show()