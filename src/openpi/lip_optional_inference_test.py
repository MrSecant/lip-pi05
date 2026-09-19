import dataclasses

import jax
import numpy as np
import pytest

from openpi.lip_inference_test import observation, policy


def test_no_visual_boundary_omits_features(policy):
    cfg = dataclasses.replace(policy.config, use_lip_visual=False)
    # Boundary validation is independent of the large transformer parameters.
    policy.config = cfg
    obs = observation(cfg)
    assert obs.get("lip_visual") is None
    converted = policy._observation(obs)
    assert converted.lip_visual is None and converted.lip_mask.shape == (1, 72)
    obs["lip_visual"] = np.zeros((1, 3, 64, 256), np.float32)
    with pytest.raises(ValueError, match="must not receive"):
        policy._observation(obs)


def test_no_visual_sampling_matches_training(policy):
    cfg = dataclasses.replace(policy.config, use_lip_visual=False)
    from openpi.lip_inference import LipPi05Policy
    runtime = LipPi05Policy(cfg.create(jax.random.key(1)), cfg, jit=True)
    inputs = observation(cfg)
    noise = np.ones((1, 32, 128), np.float32)
    got = runtime.sample(inputs, num_steps=2, seed=7, noise=noise)
    expected = runtime.model.sample_actions(jax.random.key(7), runtime._observation(inputs),
                                          num_steps=2, solver="euler", noise=noise)
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)
