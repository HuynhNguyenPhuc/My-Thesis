import trimesh
import torch
from pytorch3d.loss import chamfer_distance

def sample_points_from_mesh(glb_path: str, num_points=10000):
    """
    Load a .glb file and sample points from its surface
    """
    mesh = trimesh.load(glb_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.geometry.values())

    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)

    return torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0)

def compute_chamfer_distance(
    glb_1_path,
    glb_2_path, 
    num_points=10000
):
    """
    Compute Chamfer Distance between two .glb models.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pcd1 = sample_points_from_mesh(glb_1_path, num_points).to(device)
    pcd2 = sample_points_from_mesh(glb_2_path, num_points).to(device)
    dist, _ = chamfer_distance(pcd1, pcd2)
    return dist.item()