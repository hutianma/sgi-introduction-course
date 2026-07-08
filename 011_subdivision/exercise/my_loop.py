import numpy as np, gpytoolbox as gpy
import polyscope as ps

def my_loop(V,F,k):
    """This function performs k iterations of upsampling on the mesh V,F.
    This function returns a sparse matrix that allows the mapping of functions
    from the coarse to the fine mesh.
    """
    Vu = V.copy()
    Fu = F.copy()
    for i in range(k):
        Vu, Fu = loop_upsample(Vu, Fu)

    return Vu, Fu


def loop_upsample(V,F):
    n = V.shape[0]
    neighbors = {i: set() for i in range(n)}
    opposites = {}

    for f in F:
        v1, v2, v3 = f
        
        for a, b, c in [(v1, v2, v3), (v2, v3, v1), (v3, v1, v2)]:
            edge = tuple(sorted((a, b)))
            if edge not in opposites:
                opposites[edge] = []
            
            opposites[edge].append(c)

            neighbors[a].add(b)
            neighbors[b].add(a)

    boundary_edges = set()
    interior_edges = set()
    boundary_vertices = set()
    boundary_neighbors = {i: set() for i in range(n)}
    
    for edge, opp in opposites.items():
        a, b = edge

        if len(opp) == 1:
            boundary_edges.add(edge)

            boundary_vertices.add(a)
            boundary_vertices.add(b)
            boundary_neighbors[a].add(b)
            boundary_neighbors[b].add(a)

        elif len(opp) == 2:
            interior_edges.add(edge)
    
    interior_vertices = set(range(n)) - boundary_vertices

    new_vertices = []

    for i in range(n):
        if i in boundary_vertices:
            b_neighbors = list(boundary_neighbors[i])
            j, k = b_neighbors
            new_pos = 0.75 * V[i] + 0.125 * V[j] + 0.125 * V[k]

        else:
            valence = len(neighbors[i])

            beta = (1 / valence) * ( 
                5/8 - (3/8 + 1/4 * np.cos(2 * np.pi / valence))**2
            )

            new_pos = (1 - valence * beta) * V[i]

            for j in neighbors[i]:
                new_pos += beta * V[j]
        
        new_vertices.append(new_pos.tolist())

    edge_to_new_vertex = {}
    for edge, opp in opposites.items():
        a, b = edge
        if edge in boundary_edges:
            new_pos = 0.5 * (V[a] + V[b])

        else:
            c, d = opp
            new_pos = 3/8*V[a] + 3/8*V[b] + 1/8*V[c] + 1/8*V[d]

        new_index = len(new_vertices)
        edge_to_new_vertex[edge] = new_index
        new_vertices.append(new_pos.tolist())

    new_faces = []

    for f in F:
        v1, v2, v3 = f

        v12 = edge_to_new_vertex[tuple(sorted((v1, v2)))]
        v23 = edge_to_new_vertex[tuple(sorted((v2, v3)))]
        v31 = edge_to_new_vertex[tuple(sorted((v3, v1)))]

        new_faces.append([v1, v12, v31])
        new_faces.append([v2, v23, v12])
        new_faces.append([v3, v31, v23])
        new_faces.append([v12, v23, v31])

    Vu = np.array(new_vertices, dtype=float)
    Fu = np.array(new_faces, dtype=int)

    return Vu, Fu


V,F = gpy.read_mesh('/Users/huyufan/Documents/GitHub/sgi-introduction-course/011_subdivision/data/mug.obj')
Vu, Fu = my_loop(V, F, 4)
ps.init()
ps_mug = ps.register_surface_mesh("loop subdivided mug", Vu, Fu, smooth_shade=True)
ps.show()