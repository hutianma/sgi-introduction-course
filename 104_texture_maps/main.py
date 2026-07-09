import polyscope as ps
import numpy as np
from PIL import Image
from common.mesh import Mesh
from common.utils import plot_uvs, compute_distortion
from exercise.triangle_soup_parameterization import triangle_soup_parameterization

if __name__ == "__main__":
    ps.init()

    # Load in the mesh and texture map
    mesh = Mesh("104_texture_maps/data/cube.obj", torch=True)
    texture_map = Image.open("104_texture_maps/data/uv_grid.png")
    # normalize the texture map values to be between 0 and 1
    texture_map = np.asarray(texture_map)[:, :, :3] / 255

    
    # Compute the parameterization of the mesh
    vt, ft = triangle_soup_parameterization(mesh)

    # Visualize the parameterization in 2D (useful for debugging)
    plot_uvs(
        "/Users/huyufan/Documents/GitHub/sgi-introduction-course/104_texture_maps/data/uv_grid.png",
        vt,
        ft,
        texture_map,
        "UV Parameterization Visualization",
        linewidth=2
    )

    # Test the distortion of our parameterization
    # compute_distortion() returns the singular values of the jacobian of the
    # transformation for each triangle
    D = compute_distortion(
        mesh.vertices.cpu().numpy(),
        mesh.faces.cpu().numpy(),
        vt[ft].view(-1, 2).cpu().numpy()
    )

    # Ensure that the parameterization is angle-preserving and relative area-preserving
    # All singular values of all jacobians should equal a single value
    # (up to numerical precision)
    assert np.allclose(D, D[0, 0], atol=1e-3), \
        "Parameterization is not both angle-preserving and relative area-preserving."

    # Visualize the parameterization
    ps_mesh = ps.register_surface_mesh(
        "mesh",
        mesh.vertices.cpu().numpy(),
        mesh.faces.cpu().numpy(),
        edge_width=1,
        material='clay',
    )
    ps_mesh.add_parameterization_quantity(
        "soup_param",
        vt.cpu().numpy(),
        defined_on='corners',
        enabled=True
    )
    ps_mesh.add_color_quantity("texture", texture_map, 
                            defined_on='texture', param_name="soup_param", 
                            enabled=True)
    ps.show()
