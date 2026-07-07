import gpytoolbox as gpy, numpy as np

def my_angle_defect(V,F):
    """
    Compute the angle defect per vertex on the mesh V,F
    """

    angle_defect = np.zeros(V.shape[0])

    for f in F:
        i, j, k = f[0], f[1], f[2]
        for a, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
            u = V[b] - V[a]
            w = V[c] - V[a]
            angle = np.arccos(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)))
            angle_defect[a] += angle

    return angle_defect

V, F = gpy.read_mesh("/Users/huyufan/Documents/GitHub/sgi-introduction-course/009_curvature/data/bunny.obj")
angle_defect = my_angle_defect(V, F)
print(angle_defect)
