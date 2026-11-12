import math

import torch

from lmmviz.train_lab.losses import AlgoConfig, group_advantages, k3_kl, objective, soft_overlong_penalty


def test_advantages_zero_mean_unit_std():
    a = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(a, torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-5)


def test_on_policy_grpo_equals_mean_advantage():
    # r = 1 everywhere: no clipping, objective = mean of advantages (=0); gradient wrt logp = A/|y|
    G, L = 4, 3
    logp = torch.full((G, L), -1.0, requires_grad=True)
    o = objective(logp, logp.detach(), torch.ones(G, L), torch.tensor([1.0, 0, 0, 1]), AlgoConfig.preset("grpo"))
    assert abs(float(o["objective"])) < 1e-6 and float(o["clip_fraction"]) == 0
    o["objective"].backward()
    assert torch.allclose(logp.grad[0], torch.full((L,), 1.0 / L / G), atol=1e-5)


def test_grpo_clip_hand_computed():
    # one positive-advantage token with ratio 1.5 > 1.2: surrogate = 1.2 * A
    logp_old = torch.zeros(2, 1)
    logp = torch.tensor([[math.log(1.5)], [0.0]])
    o = objective(logp, logp_old, torch.ones(2, 1), torch.tensor([1.0, 0.0]), AlgoConfig.preset("grpo"))
    A = float(o["advantages"][0])
    assert abs(A - 1.0) < 1e-5
    assert abs(float(o["per_token_objective"][0, 0]) - 1.2 * A) < 1e-5
    assert float(o["clipped"][0, 0]) == 1.0


def test_dapo_clip_higher():
    logp = torch.tensor([[math.log(1.25)], [0.0]])
    o = objective(logp, torch.zeros(2, 1), torch.ones(2, 1), torch.tensor([1.0, 0.0]), AlgoConfig.preset("dapo"))
    assert float(o["clipped"][0, 0]) == 0.0  # 1.25 < 1.28: not clipped under clip-higher


def test_gspo_sequence_ratio_is_geometric_mean():
    logp_old = torch.zeros(1, 2)
    logp = torch.log(torch.tensor([[2.0, 0.5]]))
    o = objective(logp, logp_old, torch.ones(1, 2), torch.tensor([1.0]), AlgoConfig.preset("gspo"))
    assert abs(float(o["seq_ratio"][0]) - 1.0) < 1e-6  # sqrt(2*0.5) = 1


def test_sapo_gate_weight_peaks_at_one():
    logp_old = torch.zeros(2, 3)
    logp = torch.log(torch.tensor([[1.0, 1.5, 0.5], [1.0, 1.0, 1.0]]))
    o = objective(logp, logp_old, torch.ones(2, 3), torch.tensor([1.0, 0.0]), AlgoConfig.preset("sapo"))
    w = o["gate_weight"][0]
    assert abs(float(w[0]) - 1.0) < 1e-6 and float(w[1]) < 1 and float(w[2]) < 1
    # f(1) * A == A for both temperatures (4/τ · σ(0) = 2/τ ... times τ-scaled gradient); check gradient = A at r=1
    # on-policy (r=1) SAPO gradient equals the plain policy gradient: f'(1)=1  ->  dJ/dlogp = A / (|y| G)
    lp = torch.zeros(2, 1, requires_grad=True)
    o2 = objective(lp, torch.zeros(2, 1), torch.ones(2, 1), torch.tensor([1.0, -1.0]), AlgoConfig("sapo", std_normalize=False))
    o2["objective"].backward()
    assert torch.allclose(lp.grad[:, 0], torch.tensor([0.5, -0.5]), atol=1e-5)


def test_k3_kl_nonneg_and_zero_at_equal():
    lp = torch.tensor([-1.0, -2.0])
    assert torch.allclose(k3_kl(lp, lp), torch.zeros(2))
    assert (k3_kl(lp, lp + 0.3) >= 0).all()


def test_overlong():
    assert soft_overlong_penalty(100) == 0 and soft_overlong_penalty(20000) == -1
    assert abs(soft_overlong_penalty(14336) - (-0.5)) < 1e-9
