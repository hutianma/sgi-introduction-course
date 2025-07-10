import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from igl import grad

def trivial_parameterization(triangles):
    # Local basis origin is the first vertex in each triangle
    origins = triangles[:, 0]

    # Get the normal of the plane defined by the triangle
    e1 = triangles[:, 1] - origins
    e2 = triangles[:, 2] - origins
    normal = torch.cross(e1, e2, dim=1)
    normal /= torch.linalg.norm(normal, dim=1).unsqueeze(1)

    # We will use e1 as the first basis. Second basis will be cross between e1 and the normal
    basis1 = e1 / torch.linalg.norm(e1, dim=1).unsqueeze(1)
    basis2 = torch.cross(normal, basis1, dim=1)
    basis2 /= torch.linalg.norm(basis2, dim=1).unsqueeze(1)

    # First vertex is the origin. Project the other two vertices onto the local bases.
    # Note that in PyTorch `torch.dot()` only works with 1D tensors so we instead use
    # einsum notation to compute the dot product.
    uv0 = torch.zeros(triangles.shape[0], 2).to(triangles.device)
    uv1 = torch.stack((
        torch.einsum('bi,bi->b', e1, basis1),
        torch.einsum('bi,bi->b', e1, basis2)
    ), dim=1)
    uv2 = torch.stack((
        torch.einsum('bi,bi->b', e2, basis1),
        torch.einsum('bi,bi->b', e2, basis2)
    ), dim=1)
    uvs = torch.stack((uv0, uv1, uv2), dim=1)

    return uvs, origins, basis1, basis2

def plot_uvs(savefile, vt, ft, img,
                 name, linewidth=1,
                 xmin=0, xmax=1, ymin=0, ymax=1,
            ):
    """ Plot UV overlaid on a texture image

    Args:
        savefile (str): the path to save the image
        vt (torch.tensor): V x 2 array of UV coordinates
        ft (torch.tensor): F x 3 integer array of face indices
        img (torch.tensor): a texture image
        name (str): the title of the plot
        linewidth (float): the width of the triangle edges
        xmin (float): the minimum x value of the plot
        xmax (float): the maximum x value of the plot
        ymin (float): the minimum y value of the plot
        ymax (float): the maximum y value of the plot
    
    Returns:
        None
    """
    # First center the predicted vertices
    tris = Triangulation(vt[:, 0], vt[:, 1], triangles=ft)
    fig, axs = plt.subplots(figsize=(10, 10))

    # Plot image
    axs.imshow(img, origin='upper', extent=[xmin, xmax, ymin, ymax])

    # plot ours
    axs.set_title(name, fontsize=24)
    axs.triplot(tris, 'k-', linewidth=linewidth)

    plt.axis('off')
    axs.axis('equal')
    plt.savefig(savefile)

def get_jacobian(vs, fs, uvmap):
    """ Get jacobian of mesh given an input UV map

    Args:
        vs (np.ndarray): V x 3 array of vertex positions
        fs (np.ndarray): F x 3 integer array of face indices
        uvmap (np.ndarray): V x 2 array of UV coordinates

    Returns:
        J (np.array): F x 3 x 2 array of jacobians
    """
    G = grad(vs, fs)

    # NOTE: currently gradient is organized as X1, X2, X3, ... Y1, Y2, Y3, ... Z1, Z2, Z3
    # Reorder rows: (X1,Y1,Z1), (X2,Y2,Z2), ...
    F3 = G.shape[0]
    F = F3 // 3

    row_order = np.empty(F3, dtype=int)
    row_order[::3] = np.arange(F)
    row_order[1::3] = np.arange(F, 2*F)
    row_order[2::3] = np.arange(2*F, 3*F)

    G_reordered = G[row_order]

    J = (G_reordered @ uvmap).reshape(-1, 3, 2)
    return J

def compute_distortion(vertices, faces, uvs):
    """ Compute the distortion of the parameterization

    Args:
        vertices (np.array): V x 3 array of vertex positions
        faces (np.array): F x 3 integer array of face indices
        uvs (np.array): F*3 x 2 array of UV coordinates
    
    Returns:
        S (np.array): the singular values of the jacobian

    """
    corner_vertices = vertices[faces].reshape(-1, 3)
    corner_faces = np.arange(faces.shape[0]*3).reshape(-1, 3)
    J = get_jacobian(corner_vertices, corner_faces, uvs)
    _, S, _ = np.linalg.svd(J)
    return S

def generate_checkerboard_texture(image_size=512, num_checks=8, color1=(158, 190, 201), color2=(255, 255, 255)):
    """
    Generate a checkerboard texture image.

    Args:
        image_size (int): Size of the texture (image_size x image_size pixels)
        num_checks (int): Number of checks along one axis
        color1 (tuple): RGB color for one set of checks (default pink)
        color2 (tuple): RGB color for the other set of checks (default white)

    Returns:
        PIL.Image: The generated checkerboard image
    """
    # Create grid of UV coordinates [0, 1)
    uv = np.linspace(0, num_checks, image_size, endpoint=False)
    u, v = np.meshgrid(uv, uv)

    # Compute checker pattern: (floor(u) + floor(v)) % 2
    checks = ((np.floor(u) + np.floor(v)) % 2).astype(np.uint8)

    # Create RGB image
    img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img_array[checks == 0] = color1
    img_array[checks == 1] = color2

    # Normalize the image array to be between 0 and 1
    img_array = img_array.astype(np.float32) / 255.0

    return img_array
