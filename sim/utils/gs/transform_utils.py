import torch
import numpy as np
import kornia
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def setup_camera(w, h, k, w2c, near=0.01, far=100.0, bg=[0, 0, 0], z_threshold=0.2, sh_degree=0, device='cuda'):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=w2c.to(device),
        projmatrix=full_proj.to(device),
        sh_degree=sh_degree,
        campos=cam_center.to(device),
        prefiltered=False,
        z_threshold=z_threshold,
    )
    return cam


def Rt_to_w2c(R, t):
    c2w = np.concatenate([np.concatenate([R, t.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    w2c = np.linalg.inv(c2w)
    return w2c


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float32)


def rot_mat_to_quat(rot_mat):
    w = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
    x = (rot_mat[2, 1] - rot_mat[1, 2]) / (4 * w)
    y = (rot_mat[0, 2] - rot_mat[2, 0]) / (4 * w)
    z = (rot_mat[1, 0] - rot_mat[0, 1]) / (4 * w)
    return np.array([w, x, y, z], dtype=np.float32)


def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, weights_indices=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, k)
    # indices: (n_bones,)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, k_bones)
    # weights_indices: (n_particles, k_bones)

    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        ### sparsified weights
        dist = torch.norm(xyz[:, None] - bones, dim=-1)  # (n_particles, n_bones)
        _, indices = torch.topk(dist, 5, dim=-1, largest=False)
        bones_selected = bones[indices]  # (N, k, 3)
        dist = torch.norm(bones_selected - xyz[:, None], dim=-1)  # (N, k)
        weights = 1 / (dist + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
        weights_indices = indices

    assert weights_indices is not None
    assert weights_indices.shape[0] == weights.shape[0] == n_particles
    assert weights_indices.shape[1] == weights.shape[1]  # the indices (wrt bones) where the weights are non zero for each particle
    k_bones = weights_indices.shape[1]  # number of bones that each particle is attached to
    
    # Compute the transformed particles
    xyz_transformed = torch.zeros((n_particles, k_bones, 3), device=device)

    bones_picked = bones[weights_indices]  # (n_particles, k_bones, 3)
    bone_transforms_picked = bone_transforms[weights_indices]  # (n_particles, k_bones, 4, 4)

    xyz_transformed = xyz[:, None] - bones_picked  # (n_particles, k_bones, 3)
    xyz_transformed = (bone_transforms_picked[:, :, :3, :3] @ xyz_transformed[:, :, :, None]).squeeze(-1)  # (n_particles, k_bones, 3)
    xyz_transformed = xyz_transformed + bone_transforms_picked[:, :, :3, 3] + bones_picked  # (n_particles, k_bones, 3)
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_bones, 4)
        base_quats_picked = base_quats[weights_indices]  # (n_particles, k_bones, 4)
        quats = (base_quats_picked * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights
