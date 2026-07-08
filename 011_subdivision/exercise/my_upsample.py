import numpy as np, gpytoolbox as gpy
import polyscope as ps

def my_upsample(V,F,k):
    """This function performs k iterations of upsampling on the mesh V,F
    """
    Vu = V.copy()
    Fu = F.copy()
    for i in range(k):
        Vu, Fu = up_sampling(Vu, Fu)
    
    return Vu, Fu

def up_sampling(V,F):
    new_vertices = V.tolist()
    new_faces = []
    edge_to_new_vertex = {}

    def find_midpoint(v1, v2):
        edge = tuple(sorted((v1, v2)))

        if edge not in edge_to_new_vertex:
            new_edge = (V[v1] + V[v2]) /2
            edge_to_new_vertex[edge] = len(new_vertices)
            new_vertices.append(new_edge)

        return edge_to_new_vertex[edge]

    for face in F:
        v1, v2, v3 = face
        new_faces.append([v1, find_midpoint(v1, v2), v3])
        new_faces.append([v2, find_midpoint(v2, v3), v1])
        new_faces.append([v3, find_midpoint(v3, v1), v2])

    return np.array(new_vertices), np.array(new_faces)


V,F = gpy.read_mesh('/Users/huyufan/Documents/GitHub/sgi-introduction-course/011_subdivision/data/mug.obj')
Vu, Fu = my_upsample(V, F, 4)
ps.init()
ps_mug = ps.register_surface_mesh("upsampled mug", Vu, Fu, smooth_shade=True)
ps.show()