"""Recurrent PPO learner (Stage 2, first slice).

One parameter-shared actor-critic network drives every agent; each agent has its own recurrent (GRU) state,
which is reset whenever a slot holds a newly born agent (info["born"]). Reward is the environment's own
homeostatic reward only (HARD RULE 1): nothing here adds or reshapes it.

    obs grid (11x11x16 uint8) -> conv(32) -> conv(64, stride 2) -> fc 256 ┐
    self(13) + neighbours(4x9) + heard(8) -> fc 64                         ├-> fc 256 -> GRU 256 -> policy logits, value
                                                                           ┘
`recurrent=False` replaces the GRU by a feed-forward layer (memoryless ablation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(m, gain=np.sqrt(2)):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, obs_shapes: dict, hidden: int = 256, recurrent: bool = True):
        super().__init__()
        gh, gw, gc = obs_shapes["grid"]
        vec = int(np.prod(obs_shapes["self"]) + np.prod(obs_shapes["neighbors"]) + np.prod(obs_shapes["heard"]))
        self.conv = nn.Sequential(_init(nn.Conv2d(gc, 32, 3, padding=1)), nn.ReLU(),
                                  _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, gc, gh, gw)).shape[1]
        self.fc_grid = _init(nn.Linear(n_flat, 256))
        self.fc_vec = _init(nn.Linear(vec, 64))
        self.fc = _init(nn.Linear(320, hidden))
        self.recurrent = recurrent
        self.core = nn.GRUCell(hidden, hidden) if recurrent else _init(nn.Linear(hidden, hidden))
        self.pi = _init(nn.Linear(hidden, n_actions), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)
        self.hidden = hidden

    def encode(self, grid, selff, nb, heard):
        g = grid.float().permute(0, 3, 1, 2) / 255.0
        x = F.relu(self.fc_grid(self.conv(g)))
        v = F.relu(self.fc_vec(torch.cat([selff, nb.flatten(1), heard], 1)))
        return F.relu(self.fc(torch.cat([x, v], 1)))

    def forward_seq(self, grid, selff, nb, heard, h0, reset):
        """A whole rollout for B agents at once: inputs [T, B, ...]. The encoder runs as one batch; only the
        GRU is stepped through time (with memory resets for new agents). Returns logits [T,B,A], values [T,B]."""
        T, B = reset.shape
        x = self.encode(grid.flatten(0, 1), selff.flatten(0, 1), nb.flatten(0, 1), heard.flatten(0, 1)).view(T, B, -1)
        if self.recurrent:
            h, zs = h0, []
            for t in range(T):
                h = self.core(x[t], h * (1.0 - reset[t]).unsqueeze(1))
                zs.append(h)
            z = torch.stack(zs)
        else:
            z = F.relu(self.core(x))
        return self.pi(z), self.v(z).squeeze(-1)

    def step(self, grid, selff, nb, heard, h, reset):
        """One tick for a batch of agents. `reset` (float 0/1) zeroes the memory of newly born agents."""
        x = self.encode(grid, selff, nb, heard)
        if self.recurrent:
            h = self.core(x, h * (1.0 - reset).unsqueeze(1))
            z = h
        else:
            z = F.relu(self.core(x))
            h = torch.zeros_like(h)
        return self.pi(z), self.v(z).squeeze(1), h


@dataclass
class PPOConfig:
    rollout: int = 128          # ticks per rollout (per agent)
    gamma: float = 0.995        # ~200-tick horizon (1.7 sim days)
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 4        # split by agent (keeps sequences intact for the GRU)
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 256
    recurrent: bool = True


class Rollout:
    """Fixed-size storage of one rollout for N agent slots."""

    def __init__(self, T, N, obs_shapes):
        self.grid = np.zeros((T, N) + tuple(obs_shapes["grid"]), np.uint8)
        self.selff = np.zeros((T, N) + tuple(obs_shapes["self"]), np.float32)
        self.nb = np.zeros((T, N) + tuple(obs_shapes["neighbors"]), np.float32)
        self.heard = np.zeros((T, N) + tuple(obs_shapes["heard"]), np.float32)
        self.reset = np.zeros((T, N), np.float32)      # memory reset before acting at t (new agent in slot)
        self.alive = np.zeros((T, N), np.float32)      # slot held a live agent at t (loss mask)
        self.action = np.zeros((T, N), np.int64)
        self.logp = np.zeros((T, N), np.float32)
        self.value = np.zeros((T, N), np.float32)
        self.reward = np.zeros((T, N), np.float32)
        self.done = np.zeros((T, N), np.float32)       # the acting agent died during t (no bootstrap)


def gae(r: Rollout, last_value, cfg: PPOConfig):
    T = r.reward.shape[0]
    adv = np.zeros_like(r.reward)
    nxt_adv = np.zeros(r.reward.shape[1], np.float32)
    nxt_v = last_value
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - r.done[t]
        delta = r.reward[t] + cfg.gamma * nxt_v * nonterm - r.value[t]
        nxt_adv = delta + cfg.gamma * cfg.lam * nonterm * nxt_adv
        adv[t] = nxt_adv
        nxt_v = r.value[t]
    return adv, adv + r.value


class PPO:
    def __init__(self, n_actions, obs_shapes, cfg: PPOConfig, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = ActorCritic(n_actions, obs_shapes, cfg.hidden, cfg.recurrent).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.rng = np.random.default_rng(seed)        # minibatch order (never the global numpy RNG)

    def t(self, a, dtype=None):
        x = torch.from_numpy(np.ascontiguousarray(a))
        return x.to(self.device, dtype) if dtype else x.to(self.device)

    @torch.no_grad()
    def act(self, obs, h, reset, greedy=False):
        logits, v, h = self.net.step(self.t(obs["grid"]), self.t(obs["self"]), self.t(obs["neighbors"]),
                                     self.t(obs["heard"]), h, self.t(reset))
        probs = torch.softmax(logits.float().cpu(), 1)
        a = probs.argmax(1) if greedy else torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
        logp = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-12)
        return a.numpy(), logp.numpy(), v.float().cpu().numpy(), h, probs.numpy()

    def update(self, r: Rollout, h0, last_value):
        cfg = self.cfg
        adv, ret = gae(r, last_value, cfg)
        T, N = r.reward.shape
        idx = np.arange(N)
        stats = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0, n=0)
        tt = lambda a, d=None: self.t(a, d)  # noqa: E731
        for _ in range(cfg.epochs):
            self.rng.shuffle(idx)
            for mb in np.array_split(idx, cfg.minibatches):
                logits, values = self.net.forward_seq(tt(r.grid[:, mb]), tt(r.selff[:, mb]), tt(r.nb[:, mb]),
                                                      tt(r.heard[:, mb]), h0[mb], tt(r.reset[:, mb]))
                mask = tt(r.alive[:, mb])
                a = tt(r.action[:, mb])
                logp_old = tt(r.logp[:, mb])
                A = tt(adv[:, mb])
                R = tt(ret[:, mb])
                m = mask.sum().clamp(min=1.0)
                A = (A - (A * mask).sum() / m) / (torch.sqrt((((A - (A * mask).sum() / m) ** 2) * mask).sum() / m) + 1e-8)
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(2, a.unsqueeze(2)).squeeze(2)
                ratio = torch.exp(logp - logp_old)
                pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A)
                vf = 0.5 * (values - R) ** 2
                ent = -(logp_all.exp() * logp_all).sum(-1)
                loss = ((pg + cfg.vf_coef * vf - cfg.ent_coef * ent) * mask).sum() / m
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pg"] += float((pg * mask).sum() / m)
                    stats["vf"] += float((vf * mask).sum() / m)
                    stats["ent"] += float((ent * mask).sum() / m)
                    stats["kl"] += float((((ratio - 1) - torch.log(ratio)) * mask).sum() / m)
                    stats["clipfrac"] += float((((ratio - 1).abs() > cfg.clip).float() * mask).sum() / m)
                    stats["n"] += 1
        n = max(1, stats.pop("n"))
        return {k: v / n for k, v in stats.items()}

    def save(self, path, extra=None):
        torch.save(dict(net=self.net.state_dict(), opt=self.opt.state_dict(), cfg=self.cfg.__dict__,
                        gen=self.gen.get_state(), rng=self.rng.bit_generator.state, **(extra or {})), path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        if "gen" in ck:
            self.gen.set_state(ck["gen"])
        if "rng" in ck:
            self.rng.bit_generator.state = ck["rng"]
        return ck
