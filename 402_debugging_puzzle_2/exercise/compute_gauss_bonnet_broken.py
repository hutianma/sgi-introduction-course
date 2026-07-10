import os
import numpy as np
import gpytoolbox as gpy

def compute_total_curvature(V, F):

    '''
    Computes the sum of the (signed) Gaussian curvature at every vertex of the mesh
    '''

    # [F,3,3] tensor of corner positions for each triangle
    corner_pos = V[F,:]

    # [F,3,3] tensors of the two edge-vectors emanating from each corner
    vecA = np.roll(corner_pos, 1, axis=1) - corner_pos
    vecB = np.roll(corner_pos, 2, axis=1) - corner_pos

    # "cosine rule" to compute corner angles
    # A dot B = |A||B|cos(theta) --> theta = arccos((A dot B) / |A||B|)
    normA = np.linalg.norm(vecA, axis=-1) # [F,3]
    normB = np.linalg.norm(vecB, axis=-1) # [F,3]
    dot_prod = np.sum(vecA * vecB, axis=-1) # [F,3]

    cos_angles = dot_prod / (normA * normB)

    print("NaNs in V:", np.isnan(V).sum())
    print("NaNs in corner_pos:", np.isnan(corner_pos).sum())
    print("NaNs in vecA:", np.isnan(vecA).sum())
    print("NaNs in vecB:", np.isnan(vecB).sum())
    print("NaNs in normA:", np.isnan(normA).sum())
    print("NaNs in normB:", np.isnan(normB).sum())
    print("NaNs in dot product:", np.isnan(dot_prod).sum())
    print("NaNs in cosine:", np.isnan(cos_angles).sum())

    denominator = normA * normB
    cos_angles = dot_prod / denominator

    bad_divide = ~np.isfinite(cos_angles)

    print("bad divide locations:", np.argwhere(bad_divide))
    print("normA:", normA[bad_divide])
    print("normB:", normB[bad_divide])
    print("denominator:", denominator[bad_divide])
    print("dot product:", dot_prod[bad_divide])

    finite_cos = np.isfinite(cos_angles)

    print(
        "finite cosine below -1:",
        cos_angles[finite_cos & (cos_angles < -1)]
    )
    print(
        "finite cosine above 1:",
        cos_angles[finite_cos & (cos_angles > 1)]
    )

    cross_norm = np.linalg.norm(np.cross(vecA, vecB), axis=-1)
    corner_angles = np.arctan2(cross_norm, dot_prod) # [F,3] tensor of corner angles in radians for each triangle


    print("NaN angles:", np.isnan(corner_angles).sum())

    # compute the total curvature
    num_verts = V.shape[0]
    total_curvature = num_verts * 2.* np.pi - np.sum(corner_angles)

    return total_curvature

def process_mesh(filename):

    '''
    This function reads a mesh file, computes its total curvature, then uses the 
    Gauss-Bonnet Theorem to determine its genus. Roughly, the genus is how many 
    'handles' the shape has in a topological sense, a sphere has 0, a torus has 1, etc.

    Everything in this function is correct. 
    '''
    
    print(f"\n === Processing mesh {filename}")

    V, F = gpy.read_mesh(filename)
    
    print(f"  {V.shape[0]} verts   {F.shape[0]} faces")

    total_curvature = compute_total_curvature(V,F) #
    euler_characteristic_computed = total_curvature / (2.*np.pi) # total_curvature = 2*pi*chi
    genus_computed = (euler_characteristic_computed - 2.) / -2.  # chi = 2 - 2*genus

    print(f"  total curvature {total_curvature:.2f} = {total_curvature / np.pi:.2f} π")
    print(f"  Gauss-Bonnet Theorem says the genus is {genus_computed:.2f} (valid for closed triangle meshes only)")

# manage paths 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")

# call the function on several meshes
process_mesh(os.path.join(DATA_DIR, "sphere_good.obj"))
process_mesh(os.path.join(DATA_DIR, "spot_good.obj"))
process_mesh(os.path.join(DATA_DIR, "torus_good.obj"))
process_mesh(os.path.join(DATA_DIR, "triple_torus_good.obj"))
process_mesh(os.path.join(DATA_DIR, "sphere_bad.obj"))

import polyscope as ps

V, F = gpy.read_mesh(os.path.join(DATA_DIR, "sphere_bad.obj"))

ps.init()
ps.register_surface_mesh("sphere_bad", V, F)
ps.show()

bad_face_ids = [135, 136]
bad_vertex_ids = np.unique(F[bad_face_ids])

ps.init()
ps.register_surface_mesh("mesh", V, F)

ps.register_point_cloud(
    "degenerate-face vertices",
    V[bad_vertex_ids],
    radius=0.01,
    color=(1.0, 0.0, 0.0)
)

ps.show()