import torch
import numpy as np
import viser
import copy
from collections import deque


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port, share=False)
        self.reset_view_button = self.server.add_gui_button("Reset View")
        self.reset_state_button = self.server.add_gui_button("Reset State")

        self.need_reset = False

        self.output = {}
        self.metadata = {}

        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )

        self.fps = self.server.add_gui_text("fps", initial_value="-1", disabled=True)

        self.stop_button = self.server.add_gui_button("STOP")
        self.w_button = self.server.add_gui_button("W")
        self.a_button = self.server.add_gui_button("A")
        self.s_button = self.server.add_gui_button("S")
        self.d_button = self.server.add_gui_button("D")
        self.up_button = self.server.add_gui_button("Up")
        self.down_button = self.server.add_gui_button("Down")

        self.w_pressed = False
        self.a_pressed = False
        self.s_pressed = False
        self.d_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        @self.resolution_slider.on_update
        def _(_):
            pass

        @self.reset_view_button.on_click
        def _(_):
            for client in self.server.get_clients().values():
                client.camera.up_direction = [0.0, 0.0, 1.0]

        @self.reset_state_button.on_click
        def _(_):
            self.need_reset = True

        @self.stop_button.on_click
        def _(_):
            self.w_pressed = False
            self.a_pressed = False
            self.s_pressed = False
            self.d_pressed = False
            self.up_pressed = False
            self.down_pressed = False

        @self.w_button.on_click
        def _(_):
            self.w_pressed = True
            self.s_pressed = False

        @self.a_button.on_click
        def _(_):
            self.a_pressed = True
            self.d_pressed = False

        @self.s_button.on_click
        def _(_):
            self.s_pressed = True
            self.w_pressed = False

        @self.d_button.on_click
        def _(_):
            self.d_pressed = True
            self.a_pressed = False

        @self.up_button.on_click
        def _(_):
            self.up_pressed = True
            self.down_pressed = False

        @self.down_button.on_click
        def _(_):
            self.down_pressed = True
            self.up_pressed = False

        self.debug_idx = 0

    def set_output(self, output):
        self.output = output
    
    def get_metadata(self):
        return copy.deepcopy(self.metadata)

    @torch.no_grad()
    def update(self):
        for client in self.server.get_clients().values():
            camera = client.camera
            w2c = get_w2c(camera)
            try:
                W = self.resolution_slider.value
                H = int(self.resolution_slider.value / camera.aspect)
                focal_x = W / 2 / np.tan(camera.fov / 2)
                focal_y = H / 2 / np.tan(camera.fov / 2)

                k = np.array([[focal_x, 0, W / 2], [0, focal_y, H / 2], [0, 0, 1]])
                self.metadata = {
                    'w2c': w2c,
                    'k': k,
                    'w': W,
                    'h': H,
                }

                if self.output != {}:
                    out = self.output['image'].astype(np.uint8)
                    # print(out.shape, out.dtype, out.min(), out.max())
                else:
                    out = np.zeros((H, W, 3), dtype=np.uint8)

            except RuntimeError as e:
                print(e)
                continue

            client.set_background_image(out, format="png")
            self.debug_idx += 1

    def set_fps(self, fps):
        self.fps.value = f"{fps:.3g}"
