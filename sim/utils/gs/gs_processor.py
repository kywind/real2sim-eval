import os
import numpy as np
import torch
import plyfile
import gradio as gr
from plyfile import PlyData
import kornia
from io import BytesIO

from .transform_utils import rot_mat_to_quat, quat_mult
from .sh_utils import C0


class GSProcessor:

    def __init__(self):
        pass

    def load_phystwin(self, path, max_sh_degrees=3):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (max_sh_degrees + 1) ** 2 - 3
        features = np.zeros((xyz.shape[0], len(extra_f_names) + 3))
        features[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        for idx, attr_name in enumerate(extra_f_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features = features.reshape((features.shape[0], (max_sh_degrees + 1) ** 2), 3).permute(0, 2, 1)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        params = {
            'means3D': torch.from_numpy(xyz).to(torch.float32),
            'sh_colors': torch.from_numpy(features).to(torch.float32),
            'log_scales': torch.from_numpy(scales).to(torch.float32).repeat(1, 3),
            'unnorm_rotations': torch.from_numpy(rots).to(torch.float32),
            'logit_opacities': torch.from_numpy(opacities).to(torch.float32),
        }
        return params

    def load(self, in_dir, rot_x_minus90=False):
        # Load the point cloud
        with open(in_dir, 'rb') as f:
            plydata = plyfile.PlyData.read(f)
        
        x = torch.tensor(plydata['vertex']['x'], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(plydata['vertex']['y'], dtype=torch.float32).unsqueeze(-1)
        z = torch.tensor(plydata['vertex']['z'], dtype=torch.float32).unsqueeze(-1)
        pts = torch.cat([x, y, z], dim=-1)
        r = torch.tensor(plydata['vertex']['f_dc_0'], dtype=torch.float32)
        g = torch.tensor(plydata['vertex']['f_dc_1'], dtype=torch.float32)
        b = torch.tensor(plydata['vertex']['f_dc_2'], dtype=torch.float32)
        sh_colors = [r, g, b]
        for i in range(45):
            sh_colors.append(torch.tensor(plydata['vertex'][f'f_rest_{i}'], dtype=torch.float32))
        sh_colors = torch.stack(sh_colors, dim=-1)
        logit_opacities = torch.tensor(plydata['vertex']['opacity'], dtype=torch.float32).unsqueeze(-1)
        s0 = torch.tensor(plydata['vertex']['scale_0'], dtype=torch.float32).unsqueeze(-1)
        s1 = torch.tensor(plydata['vertex']['scale_1'], dtype=torch.float32).unsqueeze(-1)
        s2 = torch.tensor(plydata['vertex']['scale_2'], dtype=torch.float32).unsqueeze(-1)
        log_scales = torch.cat([s0, s1, s2], dim=-1)
        q0 = torch.tensor(plydata['vertex']['rot_0'], dtype=torch.float32).unsqueeze(-1)
        q1 = torch.tensor(plydata['vertex']['rot_1'], dtype=torch.float32).unsqueeze(-1)
        q2 = torch.tensor(plydata['vertex']['rot_2'], dtype=torch.float32).unsqueeze(-1)
        q3 = torch.tensor(plydata['vertex']['rot_3'], dtype=torch.float32).unsqueeze(-1)
        quats = torch.cat([q0, q1, q2, q3], dim=-1)

        # add extra rotation: make z the up axis
        if rot_x_minus90:
            rot_x_minus90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
            pts = torch.from_numpy(np.dot(rot_x_minus90, pts.cpu().numpy().T).T).to(pts.device)
            for i in range(len(quats)):
                quats[i] = torch.from_numpy(quat_mult(rot_mat_to_quat(rot_x_minus90), quats[i].cpu().numpy())).to(quats.device)

        params = {
            'means3D': pts,
            'sh_colors': sh_colors,  # (n, 48)
            'log_scales': log_scales,
            'unnorm_rotations': quats,
            'logit_opacities': logit_opacities,
        }
        return params

    def rotate(self, params, rot_mat):
        scale = np.linalg.norm(rot_mat, axis=1, keepdims=True)
        pts = params['means3D']
        quats = params['unnorm_rotations']
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot_mat_torch = torch.from_numpy(rot_mat).to(pts.device).to(pts.dtype)
        pts = pts @ rot_mat_torch.T
        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quats)
        new_rot_mat = rot_mat_torch[None] @ rot
        quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(new_rot_mat)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        params = {
            'means3D': pts,
            'sh_colors': params['sh_colors'],
            'log_scales': params['log_scales'],
            'unnorm_rotations': quats,
            'logit_opacities': params['logit_opacities'],
        }
        return params
    
    def translate(self, params, translation):
        pts = params['means3D']
        if isinstance(translation, list) or isinstance(translation, np.ndarray):
            translation = torch.tensor(translation, dtype=torch.float32).to(pts.device)
        pts = pts + translation
        params['means3D'] = pts
        return params

    def scale(self, params, scale):
        pts = params['means3D']
        if isinstance(scale, list) or isinstance(scale, np.ndarray):
            scale = torch.tensor(scale, dtype=torch.float32).to(pts.device)
        pts = pts * scale
        params['means3D'] = pts
        params['log_scales'] = torch.log(torch.exp(params['log_scales']) * scale)
        return params

    def save(self, params, save_dir):  # save to ply
        pts = params['means3D'].detach().cpu().numpy()
        colors = params['sh_colors'].reshape(params['sh_colors'].shape[0], -1).detach().cpu().numpy()
        log_scales = params['log_scales'].detach().cpu().numpy()
        quats = params['unnorm_rotations'].detach().cpu().numpy()
        logit_opacities = params['logit_opacities'].detach().cpu().numpy()

        vertex = []
        for (v, c, s, q, o) in zip(pts, colors, log_scales, quats, logit_opacities):
            vert = (
                v[0], v[1], v[2],
                c[0], c[1], c[2],
                *(c[3:].tolist()),
                o[0],
                s[0], s[1], s[2],
                q[0], q[1], q[2], q[3],
            )
            vertex.append(vert)
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        for i in range(colors.shape[1] - 3):
            vertex_dtype.append((f'f_rest_{i}', 'f4'))
        vertex_dtype += [
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]
        vertex = np.array(vertex, dtype=vertex_dtype)
        ply_el = plyfile.PlyElement.describe(vertex, 'vertex')
        ply_data = plyfile.PlyData([ply_el], text=False)
        ply_data.write(save_dir)
    
    def save_to_splat(self, params, save_dir, center=True, rotate=True):
        pts = params['means3D'].detach().cpu().numpy()
        colors = params['sh_colors'].reshape(params['sh_colors'].shape[0], -1).detach().cpu().numpy()
        scales = torch.exp(params['log_scales']).detach().cpu().numpy()
        quats = torch.nn.functional.normalize(params['unnorm_rotations'], dim=-1).detach().cpu().numpy()
        opacities = torch.sigmoid(params['logit_opacities']).detach().cpu().numpy()
        assert str(save_dir).endswith('.splat')
        if center:
            pts_mean = np.mean(pts, axis=0)
            pts = pts - pts_mean
        buffer = BytesIO()
        for (v, c, s, q, o) in zip(pts, colors, scales, quats, opacities):
            position = np.array([v[0], v[1], v[2]], dtype=np.float32)
            scales = np.array([s[0], s[1], s[2]], dtype=np.float32)
            rot = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
            color = np.array([0.5 + C0 * c[0], 0.5 + C0 * c[1], 0.5 + C0 * c[2], o[0]])

            # rotate around x axis
            if rotate:
                rot_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
                rot_x_90 = np.linalg.inv(rot_x_90)
                position = np.dot(rot_x_90, position)
                rot = quat_mult(rot_mat_to_quat(rot_x_90), rot)

            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        with open(save_dir, "wb") as f:
            f.write(buffer.getvalue())

    def crop(self, params, bbox, invert=False):
        pts = params['means3D']
        sh_colors = params['sh_colors']
        log_scales = params['log_scales']
        quats = params['unnorm_rotations']
        logit_opacities = params['logit_opacities']

        # obtain mask
        mask = (pts[:, 0] >= bbox[0][0]) & (pts[:, 0] <= bbox[0][1]) & \
            (pts[:, 1] >= bbox[1][0]) & (pts[:, 1] <= bbox[1][1]) & \
            (pts[:, 2] >= bbox[2][0]) & (pts[:, 2] <= bbox[2][1])
        if invert:
            mask = ~mask
        pts = pts[mask]
        sh_colors = sh_colors[mask]
        log_scales = log_scales[mask]
        quats = quats[mask]
        logit_opacities = logit_opacities[mask]

        print('n_pts after bbox:', pts.shape[0])

        params = {
            'means3D': pts,
            'sh_colors': sh_colors,
            'log_scales': log_scales,
            'unnorm_rotations': quats,
            'logit_opacities': logit_opacities,
        }
        return params
    
    def apply_mask(self, params, mask):
        print('n_pts after mask:', mask.sum().item())
        return {
            'means3D': params['means3D'][mask],
            'sh_colors': params['sh_colors'][mask],
            'log_scales': params['log_scales'][mask],
            'unnorm_rotations': params['unnorm_rotations'][mask],
            'logit_opacities': params['logit_opacities'][mask],
        }

    def visualize_gs(self, gs_name_list, transform=False, merged=False, axis_on=False):
        gs_name_list = [str(name) for name in gs_name_list]
        gs_name_merged_temp = "log/gs/temp_save/temp_merged.splat"
        gs_name_list_temp = []
        if merged:
            params_list = []
            for gs_name in gs_name_list:
                params = self.load(gs_name)
                params_list.append(params)
            params = self.merge(params_list)
            if axis_on:
                params = self.add_axis(params)
            self.save_to_splat(params, gs_name_merged_temp, center=transform, rotate=transform)
        else:
            for gs_name in gs_name_list:
                assert os.path.exists(gs_name)
                params = self.load(gs_name)
                gs_name_temp = f"log/gs/temp_save/temp_{gs_name.split('/')[-1].split('.')[0]}.splat"
                gs_name_list_temp.append(gs_name_temp)
                if axis_on:
                    params = self.add_axis(params)
                self.save_to_splat(params, gs_name_temp, center=transform, rotate=transform)

        with gr.Blocks() as app:
            if merged:
                with gr.Row():
                    gs = gr.Model3D(
                    value=gs_name_merged_temp,
                    clear_color=[1.0, 1.0, 1.0, 0.0],
                    label="Merged Model",
                )
            else:
                for gs_name_temp in gs_name_list_temp:
                    with gr.Row():
                        gs = gr.Model3D(
                        value=gs_name_temp,
                        clear_color=[1.0, 1.0, 1.0, 0.0],
                        label="3D Model",
                    )
        app.launch(share=True)

    def merge(self, params_list):
        return {
            'means3D': torch.cat([params['means3D'] for params in params_list], dim=0),
            'sh_colors': torch.cat([params['sh_colors'] for params in params_list], dim=0),
            'log_scales': torch.cat([params['log_scales'] for params in params_list], dim=0),
            'unnorm_rotations': torch.cat([params['unnorm_rotations'] for params in params_list], dim=0),
            'logit_opacities': torch.cat([params['logit_opacities'] for params in params_list], dim=0),
        }

    def add_axis(self, params):
        colors = params['sh_colors']
        pts = params['means3D']
        scales = torch.exp(params['log_scales'])
        quats = torch.nn.functional.normalize(params['unnorm_rotations'], dim=-1)
        opacities = torch.sigmoid(params['logit_opacities'])
        # add axis
        axis_pts = torch.tensor([
            [0, 0, 0],
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1],
        ], dtype=torch.float32, device=params['means3D'].device)
        axis_colors = torch.tensor([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=torch.float32, device=params['means3D'].device)
        axis_scales = torch.tensor([
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ], dtype=torch.float32, device=params['means3D'].device)
        axis_quats = torch.tensor([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=torch.float32, device=params['means3D'].device)
        axis_opacities = torch.tensor([
            [1],
            [1],
            [1],
            [1],
        ], dtype=torch.float32, device=params['means3D'].device)
        if len(colors.shape) == 3:  # use sh
            axis_colors = torch.cat([(axis_colors[:, None] - 0.5) / C0, 
                torch.zeros((4, colors.shape[1] - 1, 3), dtype=torch.float32, device=params['means3D'].device)], dim=1)
        if colors.shape[-1] == 48:  # use sh
            axis_colors = torch.cat([(axis_colors - 0.5) / C0, 
                torch.zeros((4, colors.shape[1] - 3), dtype=torch.float32, device=params['means3D'].device)], dim=1)
        axis_pts = torch.cat([pts, axis_pts], dim=0)
        axis_colors = torch.cat([colors, axis_colors], dim=0)
        axis_scales = torch.cat([scales, axis_scales], dim=0)
        axis_quats = torch.cat([quats, axis_quats], dim=0)
        axis_opacities = torch.cat([opacities, axis_opacities], dim=0)
        params['means3D'] = axis_pts
        params['sh_colors'] = axis_colors
        params['log_scales'] = torch.log(axis_scales)
        params['unnorm_rotations'] = axis_quats
        params['logit_opacities'] = torch.log(axis_opacities / (1 - axis_opacities))
        return params
