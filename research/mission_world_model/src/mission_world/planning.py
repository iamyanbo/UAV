"""Bounded action-conditioned CEM. Does not read a simulator or future image.

These commands are within an experimental rate envelope, NOT certified collision
avoidance. Safety unknowns remain unknown; the simulator/evaluator is separate.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class SearchBudget:
    candidates: int = 128
    iterations: int = 3
    elites: int = 16
    segments: int = 10
    horizon: float = 2.


def candidate_rollouts(dynamics, state, actions, horizon):
    result = []
    for sequence in actions:
        current = state.copy()
        for command in sequence:
            current = dynamics.step(current, command, horizon / len(sequence))
        result.append(current)
    return np.stack(result)


def constant_control_candidates(random, mean, std, count, segments, hover):
    """Exactly the committed v1 training action domain, not unseen action chunks.

    Each branch was collected with one held collective/body-rate command.
    Ten integration segments do not constitute ten independently trained action
    variables. Search the supported four-dimensional command, then repeat it.
    """
    parameters = random.normal(mean, std, (count, 4))
    parameters[:, 0] = np.clip(parameters[:, 0], hover-.02, hover+.02)
    parameters[:, 1:3] = np.clip(parameters[:, 1:3], -.06, .06)
    parameters[:, 3] = np.clip(parameters[:, 3], -.2, .2)
    return np.repeat(parameters[:, None], segments, axis=1)


class CEMPlanner:
    def __init__(self, dynamics, model, budget=SearchBudget(), seed=7):
        self.dynamics, self.model, self.budget = dynamics, model, budget
        self.random = np.random.default_rng(seed)

    @torch.inference_mode()
    def choose(self, memory, geometry, mission, ego, vehicle_state, current_camera, mission_visual=None, mission_anchor=None, camera_intrinsics=None):
        self.model.eval()
        size = self.budget
        mean = np.array([self.dynamics.hover, 0., 0., 0.])
        std = np.array([.01, .03, .03, .1])
        records = []
        best = None
        best_score = -np.inf
        for iteration in range(size.iterations):
            commands = constant_control_candidates(self.random, mean, std, size.candidates, size.segments, self.dynamics.hover)
            futures = candidate_rollouts(self.dynamics, vehicle_state, commands, size.horizon)
            scores = []
            # Sequential microbatches preserve the candidate budget without
            # multiplying memory by 128; report actual end-to-end latency.
            for action, future in zip(commands, futures):
                future_camera = self.dynamics.camera_pose(future, current_camera, vehicle_state, 1.)
                relative = np.linalg.inv(current_camera) @ future_camera
                relative[:3, 3] /= 100
                with torch.autocast("cuda", dtype=torch.float16):
                    result = self.model(memory=memory, geometry=geometry, mission=mission, ego=ego,
                        actions=torch.as_tensor(action, device=memory.device, dtype=torch.float32)[None],
                        future_camera=torch.as_tensor(relative, device=memory.device, dtype=torch.float32).flatten()[None],
                        horizon=torch.tensor([size.horizon], device=memory.device),
                        mission_visual=mission_visual, mission_anchor=mission_anchor, camera_intrinsics=camera_intrinsics)
                # Outcome index0 is mission progress, index1 measured target view
                # fraction and index2 valid-observation fraction. Their training
                # labels and units are specified in the dataset, never invented.
                score = float(result["outcomes"][0, 0])
                scores.append(score)
            ranking = np.argsort(scores)[-size.elites:]
            elite = commands[ranking, 0]
            mean, std = elite.mean(0), elite.std(0).clip(.001, None)
            winner = int(ranking[-1])
            if scores[winner] > best_score:
                best, best_score = commands[winner].copy(), scores[winner]
            records.append(dict(iteration=iteration, best_score=float(scores[winner]), candidates=len(commands)))
        return best, records
