"""Stage 2 learner: end-to-end training step, memory reset, determinism, checkpoint round-trip."""
import numpy as np
import torch

from agents.learned import LearnedPolicy
from agents.ppo import PPO, PPOConfig, Rollout, gae
from tests.helpers import SMALL, make_env


def _train(seed, iters=2, T=8, recurrent=True):
    env = make_env(seed=1, agents=dict(SMALL["agents"], respawn=True))
    cfg = PPOConfig(rollout=T, epochs=1, minibatches=2, recurrent=recurrent)
    ppo = PPO(env.n_actions, env.obs_shapes, cfg, "cpu", seed=seed)
    N = env.N
    roll = Rollout(T, N, env.obs_shapes)
    obs, h, reset = env.observe(), torch.zeros(N, cfg.hidden), np.ones(N, np.float32)
    for _ in range(iters):
        h0 = h.clone()
        for t in range(T):
            roll.grid[t], roll.selff[t], roll.nb[t], roll.heard[t] = obs["grid"], obs["self"], obs["neighbors"], obs["heard"]
            roll.reset[t], roll.alive[t] = reset, env.alive
            a, lp, v, h, _ = ppo.act(obs, h, reset)
            obs, r, d, info = env.step(a)
            roll.action[t], roll.logp[t], roll.value[t], roll.reward[t], roll.done[t] = a, lp, v, r, d
            reset = info["born"].astype(np.float32)
        stats = ppo.update(roll, h0, np.zeros(N, np.float32))
    return ppo, stats


def test_training_loop_runs_and_is_deterministic():
    p1, s1 = _train(seed=3)
    p2, s2 = _train(seed=3)
    assert all(np.isfinite(v) for v in s1.values())
    for a, b in zip(p1.net.parameters(), p2.net.parameters()):
        assert torch.equal(a, b)
    _train(seed=3, recurrent=False)          # memoryless ablation runs too


def test_memory_reset_for_new_agents():
    env = make_env(seed=1)
    ppo = PPO(env.n_actions, env.obs_shapes, PPOConfig(), "cpu", seed=0)
    obs = env.observe()
    h = torch.ones(env.N, 256)
    reset = np.zeros(env.N, np.float32)
    reset[0] = 1.0
    t = lambda k: torch.from_numpy(obs[k])  # noqa: E731
    x = ppo.net.encode(t("grid"), t("self"), t("neighbors"), t("heard"))
    _, _, h_new = ppo.net.step(t("grid"), t("self"), t("neighbors"), t("heard"), h, torch.from_numpy(reset))
    assert torch.allclose(h_new[0], ppo.net.core(x[:1], torch.zeros(1, 256))[0], atol=1e-6)   # slot 0 forgot
    assert torch.allclose(h_new[1], ppo.net.core(x[1:2], torch.ones(1, 256))[0], atol=1e-6)   # slot 1 remembered


def test_gae_does_not_bootstrap_through_death():
    r = Rollout(2, 1, {"grid": (1, 1, 1), "self": (1,), "neighbors": (1, 1), "heard": (1,)})
    r.reward[:] = [[0.0], [-1.0]]
    r.done[:] = [[0.0], [1.0]]
    r.value[:] = [[0.5], [0.5]]
    adv, ret = gae(r, np.array([100.0], np.float32), PPOConfig(gamma=0.9, lam=1.0))
    assert np.isclose(ret[1, 0], -1.0)                          # dead: no value from beyond
    assert np.isclose(ret[0, 0], 0.9 * -1.0)


def test_checkpoint_roundtrip_and_learned_policy(tmp_path):
    ppo, _ = _train(seed=5, iters=1)
    ppo.save(tmp_path / "c.pt", dict(iter=0))
    env = make_env(seed=1, agents=dict(SMALL["agents"], respawn=True))
    pol = LearnedPolicy(tmp_path / "c.pt", env, device="cpu")
    for a, b in zip(ppo.net.parameters(), pol.ppo.net.parameters()):
        assert torch.equal(a, b)
    a = pol.act(env.observe(), env.alive)
    assert a.shape == (env.N,) and set(pol.last) == {"value", "logp", "probs"}
