import gpytoolbox as gpy, polyscope as ps, numpy as np

def plot_z_coord(V,F):

    f = V[:,2]
    ps.init()
    ps_spot = ps.register_surface_mesh("spot", V, F)
    ps_spot.add_scalar_quantity("z coordinate", f, enabled=True)
    ps.register_surface_mesh("spot", V, F, smooth_shade=True, material="candy")

    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_darkness(0.8)

    ps.set_view_projection_mode("orthographic")

    ps.show()

V, F = gpy.read_mesh("/Users/huyufan/Documents/GitHub/sgi-introduction-course/005_plotting_a_function/data/spot.obj")
plot_z_coord(V, F)