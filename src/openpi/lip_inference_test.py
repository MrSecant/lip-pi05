import dataclasses

import jax
import numpy as np
import pytest

from openpi.lip_inference import IMAGE_KEYS, LipPi05Policy
from openpi.models import siglip
from openpi.models.lip_pi05 import LipPi05Config


@pytest.fixture
def policy(monkeypatch):
    factory = siglip.Module

    def tiny_image(**kwargs):
        kwargs.update(width=32, depth=1, num_heads=4, mlp_dim=64, patch_size=(112, 112))
        return factory(**kwargs)

    monkeypatch.setattr(siglip, 'Module', tiny_image)
    cfg = LipPi05Config(paligemma_variant='dummy', action_expert_variant='dummy',
                       dtype='float32', max_token_len=4)
    return LipPi05Policy(cfg.create(jax.random.key(0)), cfg, jit=True)


def observation(cfg):
    data = jax.tree.map(np.asarray, cfg.fake_obs().to_dict())
    data['image'] = {key: np.zeros((1, 224, 224, 3), np.uint8) for key in IMAGE_KEYS}
    return data


def test_sampling_matches_training_model_and_velocity(policy):
    inputs = observation(policy.config)
    obs = policy._observation(inputs)
    noise = np.ones((1, 32, 128), np.float32)
    result = policy.sample(inputs, num_steps=1, seed=9, noise=noise)
    reference = policy.model.sample_actions(jax.random.key(9), obs, num_steps=1, noise=noise, solver='euler')
    np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)
    velocity = policy.predict_velocity(inputs, noise, np.ones(1, np.float32))
    np.testing.assert_allclose(result, noise - velocity, rtol=2e-5, atol=2e-5)
    heun = policy.sample(inputs, num_steps=2, solver='heun', seed=9, noise=noise)
    expected = policy.model.sample_actions(jax.random.key(9), obs, num_steps=2, noise=noise, solver='heun')
    np.testing.assert_allclose(heun, expected, rtol=2e-5, atol=2e-5)


def test_boundary_rejects_misalignment_and_nonfinite(policy):
    inputs = observation(policy.config)
    inputs['state'] = np.zeros((1, 14), np.float32)
    with pytest.raises(ValueError, match='must agree'):
        policy.sample(inputs)
    inputs = observation(policy.config)
    inputs['lip_visual'] = np.full_like(inputs['lip_visual'], np.nan)
    with pytest.raises(ValueError, match='finite'):
        policy.sample(inputs)


def test_boundary_rejects_wrong_inputs(policy):
    inputs = observation(policy.config)
    with pytest.raises(ValueError, match='positive integer'):
        policy.sample(inputs, num_steps=0)
    with pytest.raises(ValueError, match='Latent/noise'):
        policy.sample(inputs, noise=np.zeros((1, 32, 256)))
    inputs['image'][IMAGE_KEYS[0]] = np.zeros((1, 224, 224, 3), np.float32)
    with pytest.raises(ValueError, match='uint8 RGB'):
        policy.sample(inputs)


def test_public_model_rejects_unconverted_pytorch():
    with pytest.raises(NotImplementedError, match='LIP prefix'):
        LipPi05Config().load_pytorch(None, 'unused')
