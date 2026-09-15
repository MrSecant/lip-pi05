import dataclasses

from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import lora
from openpi.models import model as base
from openpi.models import pi0
from openpi.models import siglip
from openpi.models.lip_pi05 import LipPi05Config
from openpi.models.lip_pi05 import LipPrefix
from openpi.training.config import get_config
from openpi.training.weight_loaders import merge_lip_weights


def test_missing_lora_initialization_and_strict_backbone():
    ref = {"PaliGemma": {"llm": {"attn": {
        "w": np.zeros((2, 3), np.float32),
        "lora_a": np.full((2, 1), 0.01, np.float32),
        "lora_b": np.full((1, 3), 0.02, np.float32),
    }}}}
    old = {"PaliGemma": {"llm": {"attn": {"w": np.ones((2, 3), np.float32)}}}}
    out = merge_lip_weights(old, ref)["PaliGemma"]["llm"]["attn"]
    np.testing.assert_array_equal(out["w"], 1)
    np.testing.assert_array_equal(out["lora_a"], ref["PaliGemma"]["llm"]["attn"]["lora_a"])
    old["PaliGemma"]["llm"]["attn"]["lora_a"] = np.ones((2, 1), np.float32)
    np.testing.assert_array_equal(merge_lip_weights(old, ref)["PaliGemma"]["llm"]["attn"]["lora_a"], 1)
    old["PaliGemma"]["llm"]["attn"]["lora_a"] = np.ones((3, 1), np.float32)
    with pytest.raises(ValueError, match="lora_a"):
        merge_lip_weights(old, ref)
    with pytest.raises(ValueError, match="attn/w"):
        merge_lip_weights({}, ref)
    with pytest.raises(ValueError, match="unexpected"):
        merge_lip_weights({}, {"unexpected": {"lora_a": np.zeros((1,))}})


def hybrid_test_config():
    model = LipPi05Config(paligemma_variant="gemma_2b_lora", freeze_vision_encoder=True)
    return dataclasses.replace(get_config("pi05_lip_cucumber_100k"), model=model, freeze_filter=model.get_freeze_filter())


def test_production_full_parameter_partition():
    config = get_config("pi05_lip_cucumber_100k")
    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"
    assert not config.model.freeze_vision_encoder
    state = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(0)), nnx.Param)
    count = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(state))
    assert count == 3354671244
    assert not jax.tree.leaves(state.filter(config.freeze_filter))
    assert count == sum(int(np.prod(x.shape)) for x in jax.tree.leaves(state.filter(config.trainable_filter)))


def test_optional_hybrid_parameter_partition():
    config = hybrid_test_config()
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m"
    state = nnx.state(nnx.eval_shape(config.model.create, jax.random.key(0)), nnx.Param)
    all_params = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    trainable = flax.traverse_util.flatten_dict(state.filter(config.trainable_filter).to_pure_dict(), sep="/")
    frozen = flax.traverse_util.flatten_dict(state.filter(config.freeze_filter).to_pure_dict(), sep="/")
    assert set(trainable).isdisjoint(frozen)
    assert set(trainable) | set(frozen) == set(all_params)
    groups = {"vision": 0, "vlm_base": 0, "vlm_lora": 0, "expert": 0, "other": 0}
    for key, value in all_params.items():
        if key.startswith("PaliGemma/img/"):
            group, expected = "vision", False
        elif key.startswith("PaliGemma/llm/") and "_1" in key:
            group, expected = "expert", True
            assert "lora" not in key
        elif key.startswith("PaliGemma/llm/") and "lora" in key:
            group, expected = "vlm_lora", True
        elif key.startswith("PaliGemma/llm/"):
            group, expected = "vlm_base", False
        else:
            group, expected = "other", True
        assert (key in trainable) == expected, key
        groups[group] += int(np.prod(value.shape))
    assert all(groups.values())
    print("Hybrid parameter counts:", groups)


def test_hybrid_update_keeps_frozen_parameters_identical(monkeypatch):
    factory = siglip.Module
    original_config = gemma.get_config

    def tiny_image(**kwargs):
        kwargs.update(width=32, depth=1, num_heads=4, mlp_dim=64, patch_size=(112, 112))
        return factory(**kwargs)

    def tiny_gemma(variant):
        cfg = original_config("dummy")
        if "lora" in variant:
            cfg = dataclasses.replace(cfg, lora_configs={
                "attn": lora.LoRAConfig(rank=2, alpha=2),
                "ffn": lora.LoRAConfig(rank=2, alpha=2),
            })
        return cfg

    monkeypatch.setattr(siglip, "Module", tiny_image)
    monkeypatch.setattr(gemma, "get_config", tiny_gemma)
    config = hybrid_test_config()
    cfg = dataclasses.replace(config.model, dtype="float32", max_token_len=4)
    model = cfg.create(jax.random.key(0))
    # A freshly initialized tiny expert has zero AdaRMS residual gates, which
    # block all condition gradients. Open only the test model's gates.
    state = nnx.state(model)
    state = state.map(
        lambda path, var: var.replace(value=jnp.full_like(var.value, 0.1))
        if path[-2:] == ("Dense_0", "bias") and any("norm_1" in str(part) for part in path)
        else var
    )
    nnx.update(model, state)
    obs, target = cfg.fake_obs(), cfg.fake_act()
    before = flax.traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")

    def loss_fn(m):
        return m.compute_loss(jax.random.key(3), obs, target).mean()

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(model)
    assert np.isfinite(loss)
    flat_grads = flax.traverse_util.flatten_dict(grads.to_pure_dict(), sep="/")
    assert all(np.isfinite(g).all() for g in flat_grads.values())
    for predicate in (
        lambda k: "lora" in k,
        lambda k: k.startswith("PaliGemma/llm/") and "_1" in k,
        lambda k: k.startswith("lip_prefix/"),
        lambda k: k.startswith("action_out_proj/"),
    ):
        assert any(np.any(np.asarray(g) != 0) for k, g in flat_grads.items() if predicate(k))
    trainable = nnx.state(model, config.trainable_filter)
    nnx.update(model, jax.tree.map(lambda p, g: p - 1e-3 * g, trainable, grads))
    after = flax.traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep="/")
    for key in set(before) - set(flat_grads):
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    assert any(not np.array_equal(before[k], after[k]) for k in flat_grads if "lora" in k)
    assert any(not np.array_equal(before[k], after[k]) for k in flat_grads if "_1" in k)


def test_separate_state_and_latent_dimensions():
    cfg = LipPi05Config()
    obs, target = cfg.inputs_spec(batch_size=2)
    assert obs.state.shape == (2, 14)
    assert target.shape == (2, 32, 128)
    assert obs.lip_mask.shape == (2, 264)
    with pytest.raises(ValueError, match="latent horizon"):
        LipPi05Config(action_horizon=50)


def test_preprocess_preserves_lip_conditions():
    obs = LipPi05Config().fake_obs()
    result = base.preprocess_observation(None, obs, train=False)
    for name in ("lip_visual", "lip_tactile", "lip_proprio", "lip_mask"):
        np.testing.assert_array_equal(getattr(result, name), getattr(obs, name))
    restored = base.Observation.from_dict(obs.to_dict())
    np.testing.assert_array_equal(restored.lip_tactile, obs.lip_tactile)


def test_prefix_gradients_and_mask():
    cfg = LipPi05Config()
    obs = cfg.fake_obs()
    obs = dataclasses.replace(
        obs,
        lip_visual=jax.random.normal(jax.random.key(1), obs.lip_visual.shape),
        lip_tactile=jax.random.normal(jax.random.key(2), obs.lip_tactile.shape),
        lip_proprio=jax.random.normal(jax.random.key(3), obs.lip_proprio.shape),
    )
    prefix = LipPrefix(cfg, width=32, rngs=nnx.Rngs(0))
    assert prefix(obs).shape == (1, 264, 32)
    masked = dataclasses.replace(obs, lip_mask=jnp.zeros((1, 264), dtype=bool))
    np.testing.assert_array_equal(prefix(masked), 0)

    def objective(m):
        return jnp.sum(m(obs) * jnp.arange(32))

    grads = nnx.grad(objective)(prefix)
    for group in ("visual_proj", "tactile_proj", "proprio_out"):
        assert np.linalg.norm(np.asarray(grads[group]["kernel"].value)) > 0


def test_weight_loading_is_strict_except_new_heads():
    ref = {
        "PaliGemma": {"w": np.zeros((2, 3), np.float32)},
        "action_in_proj": {"kernel": np.zeros((128, 4), np.float32)},
        "lip_prefix": {"w": np.zeros((3, 4), np.float32)},
    }
    old = {"PaliGemma": {"w": np.ones((2, 3), np.float32)}, "action_in_proj": {"kernel": np.ones((32, 4), np.float32)}}
    out = merge_lip_weights(old, ref)
    np.testing.assert_array_equal(out["PaliGemma"]["w"], 1)
    np.testing.assert_array_equal(out["action_in_proj"]["kernel"], 0)
    with pytest.raises(ValueError, match="PaliGemma/w"):
        merge_lip_weights({}, ref)


def test_tiny_end_to_end_and_attention_direction(monkeypatch):
    factory = siglip.Module

    def tiny_image(**kwargs):
        kwargs.update(width=32, depth=1, num_heads=4, mlp_dim=64, patch_size=(112, 112))
        return factory(**kwargs)

    monkeypatch.setattr(siglip, "Module", tiny_image)
    cfg = LipPi05Config(paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32", max_token_len=4)
    model = cfg.create(jax.random.key(0))
    obs, target = cfg.fake_obs(), cfg.fake_act()
    prefix, prefix_mask, prefix_ar = model.embed_prefix(obs)
    assert prefix.shape == (1, 12 + 264 + 4, 64)
    suffix, suffix_mask, suffix_ar, _ = model.embed_suffix(obs, target, jnp.ones((1,)))
    mask = pi0.make_attn_mask(jnp.concatenate((prefix_mask, suffix_mask), 1), jnp.concatenate((prefix_ar, suffix_ar)))
    p = prefix.shape[1]
    assert not np.asarray(mask[:, :p, p:]).any()
    assert np.asarray(mask[:, p:, :p]).all()
    loss = jax.jit(lambda x: model.compute_loss(jax.random.key(1), obs, x))(target)
    assert loss.shape == (1, 32)
    assert np.isfinite(loss).all()
    for solver in ("euler", "heun"):
        result = model.sample_actions(jax.random.key(2), obs, num_steps=1, solver=solver)
        assert result.shape == (1, 32, 128)
        assert np.isfinite(result).all()

    # One actual optimizer-style update on the tiny complete model.
    def loss_fn(m):
        return m.compute_loss(jax.random.key(3), obs, target).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert np.isfinite(loss)
    before = model.action_out_proj.kernel.value.copy()
    model.action_out_proj.kernel.value -= 1e-5 * grads["action_out_proj"]["kernel"].value
    assert not np.array_equal(before, model.action_out_proj.kernel.value)
