import dataclasses

import jax
import numpy as np
import pytest

from openpi.lip_inference import LipPi05Policy
from openpi.lip_inference_test import observation, policy
from openpi.models.lip_pi05 import LipPi05Config


def test_raw_action_config_contract():
    assert LipPi05Config().target_space == 'joint_latent'
    with pytest.raises(ValueError, match='Raw-action'):
        LipPi05Config(target_space='raw_action')
    with pytest.raises(ValueError, match='target_space'):
        LipPi05Config(target_space='unknown')


def test_raw_action_sampling_matches_model(policy):
    cfg = dataclasses.replace(policy.config, target_space='raw_action', use_lip_visual=False,
                              action_dim=32, action_horizon=128, temporal_downsample=1)
    runtime = LipPi05Policy(cfg.create(jax.random.key(1)), cfg, jit=True)
    inputs = observation(cfg)
    noise = np.ones((1, 128, 32), np.float32)
    result = runtime.sample(inputs, num_steps=2, seed=7, noise=noise)
    expected = runtime.model.sample_actions(jax.random.key(7), runtime._observation(inputs),
                                          num_steps=2, solver='euler', noise=noise)
    assert result.shape == (1, 128, 32) and np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match='Latent/noise'):
        runtime.sample(inputs, noise=np.zeros((1, 32, 128), np.float32))
