import torch
import numpy as np
import time
from typing import Optional
import warp as wp


class State:
    def __init__(self, num_object_points, device):
        self.wp_x = wp.zeros((num_object_points), dtype=wp.vec3, device=device, requires_grad=False)
        self.wp_v_before_collision = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()


@wp.kernel(enable_backward=False)
def set_mesh_points(
    points: wp.array(dtype=wp.vec3),
    interpolated_points: wp.array2d(dtype=wp.vec3),
    num_dynamic_points: int,
    step: int,
):
    tid = wp.tid()
    if tid < num_dynamic_points:
        points[tid] = interpolated_points[step][tid]


@wp.kernel(enable_backward=False)
def copy_2dvec3(data: wp.array2d(dtype=wp.vec3), origin: wp.array2d(dtype=wp.vec3)):
    i, j = wp.tid()
    origin[i][j] = data[i][j]


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        x1 = x[idx1]
        v1 = v[idx1]
        x2 = x[idx2]
        v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        wp.atomic_add(f, idx1, overall_force)
        wp.atomic_sub(f, idx2, overall_force)


@wp.kernel(enable_backward=False)
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_self_elas: float,
    clamp_collide_self_fric: float,
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_self_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_self_fric
                * (1.0 + clamp_collide_self_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            if resting_collision_pairs[i][index] == True or resting_collision_pairs[index][i] == True:
                continue
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel(enable_backward=False)
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_self_elas: wp.array(dtype=float),
    collide_self_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_self_elas = wp.clamp(collide_self_elas[0], low=0.0, high=1.0)
    clamp_collide_self_fric = wp.clamp(collide_self_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_self_elas,
        clamp_collide_self_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


# Calcualte the rest state pairs
@wp.kernel(enable_backward=False)
def build_resting_collision_pairs(
    x: wp.array(dtype=wp.vec3),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index < i:
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)


# This function is not validated to be differentiable yet
@wp.kernel(enable_backward=False)
def mesh_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    collide_eef_elas: wp.array(dtype=float),
    collide_eef_fric: wp.array(dtype=float),
    dt: float,
    mesh_map: wp.array(dtype=int),
    face_map: wp.array(dtype=int),
    dynamic_velocity: wp.array(dtype=wp.vec3),
    dynamic_omega: wp.array(dtype=wp.vec3),
    step: int,
    interpolated_center: wp.array2d(dtype=wp.vec3),
    use_pusher: bool,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
    collision_forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    next_x = x0 + v0 * dt
    query = wp.mesh_query_point_sign_winding_number(
        mesh, next_x, max_dist=0.02, accuracy=3.0, threshold=0.6
    )
    if query.result:
        # Judge if this is the gripper
        if not use_pusher:
            if mesh_map[query.face] == 0:
                is_gripper = 1
            elif mesh_map[query.face] == 1:
                is_gripper = 2
            else:  # < 0: static mesh
                is_gripper = 0
        else:
            if mesh_map[query.face] >= 0:
                is_gripper = 1
            else:  # < 0: static mesh
                is_gripper = 0

        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = next_x - p
        dist = wp.length(delta) * query.sign

        if is_gripper >= 1 and not use_pusher:
            margin = 0.005  # 5mm
        else:
            margin = 0.001  # 1mm (prevent too much shape change for static mesh)

        err = dist - margin

        if err < 0.0:
            normal = wp.normalize(delta) * query.sign

            real_dynamic_velocity = wp.vec3(0.0, 0.0, 0.0)
            if is_gripper >= 1:
                if is_gripper == 1:
                    real_dynamic_velocity = dynamic_velocity[0] + wp.cross(
                        dynamic_omega[0], (x0 - interpolated_center[step][0])  # first arm, first finger
                    )
                elif is_gripper == 2:
                    real_dynamic_velocity = dynamic_velocity[1] + wp.cross(
                        dynamic_omega[0], (x0 - interpolated_center[step][0])  # first arm, second finger
                    )
                v0 = v0 - real_dynamic_velocity
                clamp_collide_elas = wp.clamp(collide_eef_elas[0], low=0.0, high=1.0)
                clamp_collide_fric = wp.clamp(collide_eef_fric[0], low=0.0, high=2.0)
            else:
                clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
                clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

            v_normal = wp.dot(v0, normal) * normal
            v_tao = v0 - v_normal
            v_normal_length = wp.length(v_normal)
            v_tao_length = wp.max(wp.length(v_tao), 1e-6)

            v_normal_new = -clamp_collide_elas * v_normal

            delta_v_normal = v_normal_new - v_normal

            a = wp.max(
                0.0,
                1.0
                - clamp_collide_fric
                * (1.0 + clamp_collide_elas)
                * v_normal_length
                / v_tao_length,
            )
            v_tao_new = a * v_tao

            next_v = v_normal_new + v_tao_new
            if is_gripper >= 1:
                next_v += real_dynamic_velocity

            if is_gripper >= 1:
                # Use new speed to judge the new collision position
                next_x = x0 + next_v * dt
                query = wp.mesh_query_point_sign_winding_number(
                    mesh, next_x, max_dist=0.02, accuracy=3.0, threshold=0.6
                )
                if query.result:
                    p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    delta = next_x - p
                    dist = wp.length(delta) * query.sign
                    err = dist - margin

                    if err < 0.0:
                        normal = wp.normalize(delta) * query.sign
                        next_x = next_x - normal * err
            else:
                next_x = next_x - normal * err

            # Only calculate the force in the normal direction
            delta_v_normal = v_normal_new - v_normal
            wp.atomic_add(collision_forces, face_map[query.face], delta_v_normal / dt)
        else:
            next_v = v0
    else:
        next_v = v0

    x_new[tid] = next_x
    v_new[tid] = next_v


@wp.kernel(enable_backward=False)
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    ground_height = 0.0
    if next_x_z < ground_height and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -(x_z - ground_height) / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


class SpringMassSystemWarp:
    def __init__(
        self,
        phystwin_cfg,
        device,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        num_object_points,
        init_spring_Y=None,
        collide_elas=None,
        collide_fric=None,
        collide_eef_elas=None,
        collide_eef_fric=None,
        collide_self_elas=None,
        collide_self_fric=None,
        init_collision_mask=None,
        init_velocities=None,
        dynamic_meshes=None,
        static_meshes=None,
        dynamic_points=None,
        use_pusher=False,
    ):
        dt = phystwin_cfg.dt
        num_substeps = phystwin_cfg.num_substeps
        spring_Y = phystwin_cfg.init_spring_Y
        dashpot_damping = phystwin_cfg.dashpot_damping
        drag_damping = phystwin_cfg.drag_damping
        collision_dist = phystwin_cfg.collision_dist
        self.device = device
        reverse_z = phystwin_cfg.reverse_z
        spring_Y_min = phystwin_cfg.spring_Y_min
        spring_Y_max = phystwin_cfg.spring_Y_max
        self_collision = phystwin_cfg.self_collision
        use_graph = phystwin_cfg.use_graph
        self.self_collision = self_collision
        self.use_pusher = use_pusher

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        self.n_springs = init_springs.shape[0]
        self.num_object_points = num_object_points
        assert num_object_points == init_vertices.shape[0]

        if self_collision:
            if init_collision_mask is None:
                init_collision_mask = torch.arange(
                    num_object_points, 
                    dtype=torch.int32, device=self.device
                )
            assert torch.unique(init_collision_mask).shape[0] > 1
            self.wp_masks = wp.from_torch(
                init_collision_mask[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (num_object_points, 500),
                dtype=wp.int32,
                device=self.device, 
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (num_object_points), dtype=wp.int32, device=self.device, requires_grad=False
            )

        # Initialize the spring system
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )

        self.wp_state = State(num_object_points, self.device)

        wp_init_vertices = wp.from_torch(
            init_vertices.contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        
        if init_velocities is None:
            wp_init_velocities = wp.zeros_like(
                wp_init_vertices, requires_grad=False
            )
        else:
            wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )
        self.set_init_state(wp_init_vertices, wp_init_velocities)

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([phystwin_cfg.collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([phystwin_cfg.collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )
        self.wp_collide_eef_elas = wp.from_torch(
            torch.tensor([phystwin_cfg.collide_eef_elas], dtype=torch.float32, device=self.device),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )
        self.wp_collide_eef_fric = wp.from_torch(
            torch.tensor([phystwin_cfg.collide_eef_fric], dtype=torch.float32, device=self.device),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )
        self.wp_collide_self_elas = wp.from_torch(
            torch.tensor(
                [phystwin_cfg.collide_self_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )
        self.wp_collide_self_fric = wp.from_torch(
            torch.tensor(
                [phystwin_cfg.collide_self_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=phystwin_cfg.collision_requires_grad,
        )

        self.set_spring_Y(init_spring_Y)
        self.set_collide(collide_elas, collide_fric)
        self.set_collide_eef(collide_eef_elas, collide_eef_fric)
        self.set_collide_self(collide_self_elas, collide_self_fric)

        # Load the static meshes
        self.all_meshes_warp = None
        if static_meshes is not None or dynamic_meshes is not None:
            vertices = []
            indices = []
            mesh_map = []
            face_map = []
            mesh_index = 0
            face_index = 0
            offset = 0

            if dynamic_meshes is not None:
                for dynamic_mesh in dynamic_meshes:
                    vertex = np.array(dynamic_mesh.vertices, dtype=np.float32)
                    index = np.array(dynamic_mesh.triangles, dtype=np.int32).flatten()
                    vertices.append(vertex)
                    indices.append(index + offset)
                    offset += vertex.shape[0]
                    mesh_map.append(
                        np.ones(int(index.shape[0] / 3), dtype=np.int32) * mesh_index
                    )
                    face_map.append(
                        np.arange(int(index.shape[0] / 3), dtype=np.int32) + face_index
                    )
                    mesh_index += 1
                    face_index += int(index.shape[0] / 3)

            mesh_index = -1

            if static_meshes is not None:
                for static_mesh in static_meshes:
                    vertex = np.array(static_mesh.vertices, dtype=np.float32)
                    index = np.array(static_mesh.triangles, dtype=np.int32).flatten()
                    vertices.append(vertex)
                    indices.append(index + offset)
                    offset += vertex.shape[0]
                    mesh_map.append(
                        np.ones(int(index.shape[0] / 3), dtype=np.int32) * mesh_index
                    )
                    face_map.append(
                        np.arange(int(index.shape[0] / 3), dtype=np.int32) + face_index
                    )
                    mesh_index -= 1
                    face_index += int(index.shape[0] / 3)

            vertices = np.concatenate(vertices, axis=0)
            indices = np.concatenate(indices, axis=0)

            self.all_meshes_warp = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                indices=wp.array(indices, dtype=int, device=self.device),
            )
            self.mesh_map = wp.array(
                np.concatenate(mesh_map, axis=0),
                dtype=int,
                device=self.device,
                requires_grad=False,
            )
            face_map = np.concatenate(face_map, axis=0)
            self.face_map = wp.array(
                face_map,
                dtype=int,
                device=self.device,
                requires_grad=False,
            )
            self.collision_forces = wp.array(
                np.zeros([len(face_map), 3], dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=False,
            )
            assert isinstance(dynamic_points, torch.Tensor)
            self.num_eefs = len(dynamic_meshes) // 2 if not self.use_pusher else len(dynamic_meshes)
            assert self.num_eefs <= 1
            self.wp_interpolated_dynamic_points = wp.from_torch(
                dynamic_points.clone().repeat(self.num_substeps, 1, 1),
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.wp_interpolated_center = wp.from_torch(
                torch.mean(dynamic_points.reshape(self.num_eefs, -1, 3), dim=1).clone().repeat(self.num_substeps, 1, 1),
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.num_dynamic_velocities = self.num_eefs * 2 if not self.use_pusher else self.num_eefs
            self.wp_dynamic_velocity = wp.zeros((self.num_dynamic_velocities), dtype=wp.vec3, device=self.device, requires_grad=False)
            self.wp_dynamic_omega = wp.zeros((self.num_eefs), dtype=wp.vec3, device=self.device, requires_grad=False)
            self.num_dynamic_points = len(dynamic_points)

        if self_collision:
            self.resting_collision_pairs = wp.zeros(
                (wp_init_vertices.shape[0], wp_init_vertices.shape[0]),
                dtype=wp.bool,
                device=self.device, 
                requires_grad=False,
            )
            self.create_resting_case()

        if use_graph:
            with wp.ScopedCapture() as capture:
                self.step()
            self.graph = capture.graph

    # Create the rest map for self-collision in frame 0
    def create_resting_case(self):
        self.collision_grid.build(self.wp_state.wp_x, self.collision_dist * 5.0)
        wp.launch(
            build_resting_collision_pairs,
            dim=self.num_object_points,
            inputs=[
                self.wp_state.wp_x,
                self.collision_dist,
                self.collision_grid.id,
                ],
            outputs=[self.resting_collision_pairs],            
        )  

    def set_init_state(self, x, v):
        if isinstance(x, torch.Tensor):
            x = wp.from_torch(
                x.contiguous(), dtype=wp.vec3, requires_grad=False
            )
        if isinstance(v, torch.Tensor):
            v = wp.from_torch(
                v.contiguous(), dtype=wp.vec3, requires_grad=False
            )

        assert (
            self.num_object_points == x.shape[0]
            and self.num_object_points == self.wp_state.wp_x.shape[0]
        )
        wp.launch(
            copy_vec3,
            dim=self.num_object_points,
            inputs=[x],
            outputs=[self.wp_state.wp_x],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_object_points,
            inputs=[v],
            outputs=[self.wp_state.wp_v],
        )

    def set_mesh_interactive(
        self,
        interpolated_dynamic_points,
        interpolated_center,
        dynamic_velocity,
        dynamic_omega,
    ):
        wp.launch(
            copy_2dvec3,
            dim=(
                interpolated_dynamic_points.shape[0],
                interpolated_dynamic_points.shape[1],
            ),
            inputs=[interpolated_dynamic_points],
            outputs=[self.wp_interpolated_dynamic_points],
        )
        wp.launch(
            copy_2dvec3,
            dim=len(interpolated_center),
            inputs=[interpolated_center],
            outputs=[self.wp_interpolated_center],
        )

        wp.launch(
            copy_vec3,
            dim=self.num_dynamic_velocities,
            inputs=[dynamic_velocity],
            outputs=[self.wp_dynamic_velocity],
        )

        wp.launch(
            copy_vec3,
            dim=self.num_eefs,
            inputs=[dynamic_omega],
            outputs=[self.wp_dynamic_omega],
        )

    def update_collision_graph(self):
        assert self.self_collision
        self.collision_grid.build(self.wp_state.wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_state.wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
                self.resting_collision_pairs,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_state.clear_forces()

            with wp.ScopedTimer("eval_springs", synchronize=False):
                # Calculate the spring forces
                wp.launch(
                    kernel=eval_springs,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_state.wp_x,
                        self.wp_state.wp_v,
                        self.wp_springs,
                        self.wp_rest_lengths,
                        self.wp_spring_Y,
                        self.dashpot_damping,
                        self.spring_Y_min,
                        self.spring_Y_max,
                    ],
                    outputs=[self.wp_state.wp_vertice_forces],
                )

            if self.self_collision:
                output_v = self.wp_state.wp_v_before_collision
            else:
                output_v = self.wp_state.wp_v_before_ground

            with wp.ScopedTimer("update_vel_from_force", synchronize=False):
                # Update the output_v using the vertive_forces
                wp.launch(
                    kernel=update_vel_from_force,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_state.wp_v,
                        self.wp_state.wp_vertice_forces,
                        self.wp_masses,
                        self.dt,
                        self.drag_damping,
                        self.reverse_factor,
                    ],
                    outputs=[output_v],
                )

            if self.self_collision:
                with wp.ScopedTimer("object_collision", synchronize=False):
                    # Update the wp_v_before_ground based on the collision handling
                    wp.launch(
                        kernel=object_collision,
                        dim=self.num_object_points,
                        inputs=[
                            self.wp_state.wp_x,
                            self.wp_state.wp_v_before_collision,
                            self.wp_masses,
                            self.wp_masks,
                            self.wp_collide_self_elas,
                            self.wp_collide_self_fric,
                            self.collision_dist,
                            self.wp_collision_indices,
                            self.wp_collision_number,
                        ],
                        outputs=[self.wp_state.wp_v_before_ground],
                    )

            # This function is not promised to be differentiable for now
            if self.all_meshes_warp is not None:
                with wp.ScopedTimer("set_mesh", synchronize=False):
                    wp.launch(
                        set_mesh_points,
                        dim=len(self.all_meshes_warp.points),
                        inputs=[
                            self.all_meshes_warp.points,
                            self.wp_interpolated_dynamic_points,
                            self.num_dynamic_points,
                            i,
                        ],
                    )
                    self.all_meshes_warp.refit()
                    self.collision_forces.zero_()
                with wp.ScopedTimer("mesh_collision", synchronize=False):
                    wp.launch(
                        kernel=mesh_collision,
                        dim=self.num_object_points,
                        inputs=[
                            self.wp_state.wp_x,
                            self.wp_state.wp_v_before_ground,
                            self.all_meshes_warp.id,
                            self.wp_collide_elas,
                            self.wp_collide_fric,
                            self.wp_collide_eef_elas,
                            self.wp_collide_eef_fric,
                            self.dt,
                            self.mesh_map,
                            self.face_map,
                            self.wp_dynamic_velocity,
                            self.wp_dynamic_omega,
                            i,
                            self.wp_interpolated_center,
                            self.use_pusher,
                        ],
                        outputs=[
                            self.wp_state.wp_x,
                            self.wp_state.wp_v_before_ground,
                            self.collision_forces,
                        ],
                    )

            with wp.ScopedTimer("integrate_ground_collision", synchronize=False):
                # Update the x and v
                wp.launch(
                    kernel=integrate_ground_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_state.wp_x,
                        self.wp_state.wp_v_before_ground,
                        self.wp_collide_elas,
                        self.wp_collide_fric,
                        self.dt,
                        self.reverse_factor,
                    ],
                    outputs=[self.wp_state.wp_x, self.wp_state.wp_v],
                )

    # Functions used to load the parameters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_eef(self, collide_eef_elas, collide_eef_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_eef_elas],
            outputs=[self.wp_collide_eef_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_eef_fric],
            outputs=[self.wp_collide_eef_fric],
        )

    def set_collide_self(self, collide_self_elas, collide_self_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_self_elas],
            outputs=[self.wp_collide_self_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_self_fric],
            outputs=[self.wp_collide_self_fric],
        )
