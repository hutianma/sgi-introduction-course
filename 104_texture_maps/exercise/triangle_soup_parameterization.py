import torch
from common.utils import trivial_parameterization
from .pack_triangles import pack_triangles

def triangle_soup_parameterization(mesh):
    """
    This function compute the "trivial" triangle soup parameterization.
    Input: mesh object containing vertices and faces
    Output: num_faces x num_vertices x 2 tensor containing the UV coordinates at each
    vertex of each face

    Suggested approach:
        (1) Independently flatten all triangles to the plane. Hint: use the approach in
            exercise 101. See the function trivial_parameterization().
        (2) Use the pack_triangles function to rescale and arrange all of the triangles
            in the unit square
        (3) Rearrange the UV coordinates (u and v are just the x and y coordinates of
            the triangles in the unit square) to get two arrays:
                vt: tensor of shape (num_faces*3, 2) that contains the UV coordinates of
                    each vertex. Note that the first dimension is num_faces*3 not
                    num_vertices because here we want UV coordinates per face and thus a
                    vertex in the original mesh that is a part of multiple faces will
                    have multiple (potentially different) UV coordinates (one for each
                    face it is contained in).
                ft: tensor of shape (num_faces, 3) that contains the indices into the vt
                    tensor for the vertices of each face in the original mesh.
    
    Args:
        mesh (Mesh): mesh object containing vertices and faces
    
    Returns:
        vt (torch.tensor): num_faces*3 x 2 tensor containing the UV coordinates of each
                           vertex
        ft (torch.tensor): num_faces x 3 tensor containing the indices into vt for the
                           vertices of each face
    """
    # Get the device (either "cpu" or "cuda")
    # Any time we create a PyTorch tensor from scratch, we need to specify the device
    # For example: new_tensor = torch.tensor([1, 2, 3]).to(device)
    device = mesh.vertices.device

    triangle_3d = mesh.vertices[mesh.faces].to(device)

    # Map all triangles to the plane independently (use `trivial_parameterization()`)
    flatten_result = trivial_parameterization(triangle_3d)
    flatten_tri = flatten_result[0]

    
    # Pack triangles into a unit sqaure
    packed_tri = pack_triangles(flatten_tri)

    # Construct the vt and ft tensors using the packed triangles
    num_faces = packed_tri.shape[0]
    vt = packed_tri.reshape(num_faces * 3, 2)
    # Hint: see torch.arange() for creating the ft tensor
    ft = torch.arange(num_faces * 3).reshape(num_faces, 3)

    return vt, ft
