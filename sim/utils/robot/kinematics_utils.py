import numpy as np
import transforms3d
import sapien.core as sapien


class KinHelper():
    def __init__(self, robot_name, sapien_env_tuple=None, headless=True):
        # load robot
        if "xarm7" in robot_name:
            urdf_path = str("assets/robots/xarm/xarm7.urdf")
            self.eef_name = 'link7'
        else:
            raise RuntimeError('robot name not supported')
        self.robot_name = robot_name

        # load sapien robot
        if sapien_env_tuple is not None:
            engine, scene, loader = sapien_env_tuple
            self.engine = engine
            self.scene = scene
        else:
            self.engine = sapien.Engine()
            self.scene = self.engine.create_scene()
            loader = self.scene.create_urdf_loader()

        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == self.eef_name:
                self.sapien_eef_idx = link_idx
                break

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        self.pcd_dict = {}
        self.tool_meshes = {}

    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def compute_ik_sapien(self, initial_qpos, cartesian, verbose=False):
        """
        Compute IK using sapien
        initial_qpos: (N, ) numpy array
        cartesian: (6, ) numpy array, x,y,z in meters, r,p,y in radians
        """
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes='sxyz')
        tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf_mat)

        active_qmask = np.array([True, True, True, True, True, True, True])
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx, 
            pose=pose,
            initial_qpos=initial_qpos, 
            active_qmask=active_qmask, 
            )
        if verbose:
            print('ik qpos:', qpos)

        # verify ik
        fk_pose = self.compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        
        if verbose:
            print('target pose for IK:', tf_mat)
            print('fk pose for IK:', fk_pose)
        
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf_mat[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf_mat[:3, :3])
        
        if pose_diff > 0.01 or rot_diff > 0.01:
            print('ik pose diff:', pose_diff)
            print('ik rot diff:', rot_diff)
            print('ik pose diff or rot diff too large. Return initial qpos.')
            return initial_qpos
        return qpos[0]
