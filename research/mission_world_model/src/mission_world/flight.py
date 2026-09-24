"""Published FiGS rigid-body/rate equations, evaluated with a documented RK4 adapter.

The derivative is imported from FiGS, not reimplemented. The native Acados IRK
solver/controller is a separate reference gate; this adapter is not its reproduction.
Coordinates are NED, state quaternion is xyzw, collective hover command is NEGATIVE.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .upstream import load_file, verify_revision


class FigsDynamics:
    def __init__(self, root):
        from casadi import Function
        root = Path(root)
        repo = root / "upstream/figs"
        verify_revision(repo, "11ad36ccf294c8423b54e4e7263a03e6a246f1b1")
        sys.path.insert(0, str(repo / "acados/interfaces/acados_template"))
        module = load_file("idea1_figs_rate", repo / "src/figs/dynamics/quadcopter_rate_model.py")
        model = module.export_model()
        self.derivative = Function("figs_derivative", [model.x, model.u, model.p], [model.f_expl_expr])
        self.frame = json.loads((root / "upstream/sousvide/configs/frames/carl.json").read_text())
        self.params = np.array([self.frame["mass"], self.frame["motor_thrust_coeff"], 0., 0., 0.])
        self.hover = -self.params[0] * 9.81 / (4 * self.params[1])
        self.dt = .01

    def step(self, state, command, duration=.1):
        state = np.asarray(state, dtype=np.float64).copy()
        command = np.asarray(command, dtype=np.float64)
        if state.shape != (10,) or command.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(command).all():
            raise ValueError("FiGS expects finite 10D state and 4D applied rate/thrust command")
        if not (-1 <= command[0] <= 0) or np.max(np.abs(command[1:])) > 2:
            raise ValueError("Command outside declared experiment envelope")
        steps = round(duration / self.dt)
        if steps <= 0 or not np.isclose(steps * self.dt, duration):
            raise ValueError("Duration must be a positive multiple of integration timestep")
        derivative = lambda x: np.asarray(self.derivative(x, command, self.params)).ravel()
        for _ in range(steps):
            a = derivative(state)
            b = derivative(state + self.dt * a / 2)
            c = derivative(state + self.dt * b / 2)
            d = derivative(state + self.dt * c)
            state += self.dt / 6 * (a + 2*b + 2*c + d)
            state[6:] /= np.linalg.norm(state[6:])
        if not np.isfinite(state).all():
            raise RuntimeError("Nonfinite dynamics")
        return state

    @staticmethod
    def camera_pose(state, initial_c2w, initial_state, scene_units_per_meter):
        """Attach initial camera extrinsics to vehicle and preserve them while moving.

        Scene is Z-up; NED is mapped through diag(1,-1,-1), as in FiGS. Camera
        calibration comes from released scene cameras, not the carl lens. Aerial
        initial orientation is therefore a documented mount adaptation.
        """
        if scene_units_per_meter <= 0:
            raise ValueError("Verified scene scale is required")
        world_axes = np.diag([1., -1., -1.])
        current_rotation = Rotation.from_quat(state[6:]).as_matrix()
        initial_rotation = Rotation.from_quat(initial_state[6:]).as_matrix()
        delta = world_axes @ current_rotation @ initial_rotation.T @ world_axes.T
        pose = np.array(initial_c2w, copy=True)
        pose[:3, :3] = delta @ initial_c2w[:3, :3]
        pose[:3, 3] += world_axes @ (state[:3] - initial_state[:3]) * scene_units_per_meter
        return pose
