import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import open3d as o3d

@torch.jit.script
def compute_intersection_union(voxel_grid1: torch.Tensor, voxel_grid2: torch.Tensor) -> torch.Tensor:
    """ 
    Compute intersection and union using PyTorch GPU operations.
    """
    intersection = torch.logical_and(voxel_grid1, voxel_grid2).sum()
    union = torch.logical_or(voxel_grid1, voxel_grid2).sum()
    return torch.stack([intersection, union])

def voxelize(mesh: o3d.geometry.TriangleMesh, voxel_size=1/64, grid_dim=64, device="cuda") -> torch.Tensor:
    """
    Convert a 3D mesh into a voxel grid tensor.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh.
        voxel_size (float): Size of each voxel.
        grid_dim (int): Size of voxel grid.
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: A 3D occupancy grid tensor on GPU.
    """
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh is empty!")
    vertices = np.asarray(mesh.vertices)
    aabb = np.stack([vertices.min(0), vertices.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = (aabb[1] - aabb[0]).max()
    vertices = (vertices - center) / scale
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5)
    )

    voxel_list = voxel_grid.get_voxels()
    if len(voxel_list) == 0:
        raise ValueError("Voxelization failed! No valid voxels found.")

    voxel_tensor = torch.zeros((grid_dim, grid_dim, grid_dim), dtype=torch.uint8, device=device)
    voxel_indices = torch.tensor([voxel.grid_index for voxel in voxel_list], dtype=torch.long, device=device)

    if (voxel_indices >= grid_dim).any() or (voxel_indices < 0).any():
        raise ValueError(f"Invalid voxel indices: {voxel_indices}")

    voxel_tensor[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

    return voxel_tensor

def compute_iou(glb_file1: str, glb_file2: str, resolution=64, device="cuda") -> float:
    """
    Compute IoU (Intersection over Union) between two 3D meshes in .glb files.

    Args:
        glb_file1 (str): Path to first .glb file.
        glb_file2 (str): Path to second .glb file.
        device (str): "cuda" or "cpu".

    Returns:
        float: IoU value.
    """
    mesh1 = o3d.io.read_triangle_mesh(glb_file1)
    mesh2 = o3d.io.read_triangle_mesh(glb_file2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(voxelize, mesh1, 1/resolution, resolution, device)
        future2 = executor.submit(voxelize, mesh2, 1/resolution, resolution, device)
        voxel_grid1 = future1.result()
        voxel_grid2 = future2.result()

    with torch.amp.autocast(device_type=device):
        iou_values = compute_intersection_union(voxel_grid1, voxel_grid2)

    intersection, union = iou_values[0].item(), iou_values[1].item()
    iou = intersection / union if union > 0 else 0.0
    return iou