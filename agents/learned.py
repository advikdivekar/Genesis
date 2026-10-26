"""A trained PPO checkpoint wrapped as a policy with the same interface as the baselines
(`act(obs, alive)`, `reset(mask)`), so it can drive the viewer and evaluation runs."""
from __future__ import annotations

import numpy as np
import torch

from agents.ppo import PPO, PPOConfig


class LearnedPolicy:
    name = "ppo"

    def __init__(self, checkpoint, env, device: str | None = None, greedy: bool = False, seed: int = 0):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = PPOConfig(**ck["cfg"])
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.ppo = PPO(env.n_actions, env.obs_shapes, cfg, device, seed=seed)
        self.ppo.net.load_state_dict(ck["net"])
        self.ppo.net.eval()
        self.greedy = greedy
        self.checkpoint = str(checkpoint)
        self.iter = ck.get("iter")
        self.h = torch.zeros(env.N, cfg.hidden, device=self.ppo.device)
        self.pending_reset = np.ones(env.N, np.float32)
        self.mode = None                      # no discrete internal mode (recorder policy_state = -1)
        self.last = {}

    def reset(self, mask=None):
        if mask is None:
            self.pending_reset[:] = 1.0
        else:
            self.pending_reset[np.asarray(mask, bool)] = 1.0

    def act(self, obs, alive):
        a, logp, v, self.h, probs = self.ppo.act(obs, self.h, self.pending_reset, greedy=self.greedy)
        self.pending_reset[:] = 0.0
        self.last = dict(value=v, logp=logp, probs=probs)   # learner internals for the flight recorder
        return a
