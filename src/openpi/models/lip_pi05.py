"""Pi0.5 policy over a frozen, task-specific LIP latent space."""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model as base
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


@dataclasses.dataclass(frozen=True)
class LipPi05Config(Pi0Config):
    pi05: bool = True
    action_dim: int = 128  # Internal expert output is a latent, never a robot state.
    action_horizon: int = 32
    state_dim: int = 14
    decoded_action_horizon: int = 128
    temporal_downsample: int = 4
    visual_views: int = 3
    visual_tokens_per_view: int = 64
    visual_dim: int = 256
    tactile_dim: int = 64
    history_frames: int = 8
    tactile_sensors: int = 4
    tactile_tokens_per_sensor: int = 2
    proprio_hidden_dim: int = 64
    visual_adapter_dim: int = 128
    use_lip_visual: bool = True
    freeze_vision_encoder: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("LIP Pi0.5 retains the native discrete current-state interface")
        if self.action_horizon * self.temporal_downsample != self.decoded_action_horizon:
            raise ValueError("latent horizon and decoded action horizon disagree")
        if self.visual_tokens_per_view != 64:
            raise ValueError("v1 feature cache uses spatial 16x16 -> 8x8 pooling")

    @property
    def lip_token_count(self):
        return (self.visual_views * self.visual_tokens_per_view if self.use_lip_visual else 0) + self.history_frames * (
            self.tactile_sensors * self.tactile_tokens_per_sensor + 1
        )

    def create(self, rng):
        return LipPi05(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path):
        raise NotImplementedError(
            "LIP prefix is not implemented by the upstream PyTorch model; use the JAX inference backend"
        )

    def get_freeze_filter(self):
        backbone_filter = super().get_freeze_filter()
        if self.freeze_vision_encoder:
            return nnx.Any(backbone_filter, nnx_utils.PathRegex("PaliGemma/img/.*"))
        return backbone_filter

    def inputs_spec(self, *, batch_size=1):
        obs, target = super().inputs_spec(batch_size=batch_size)

        def spec(shape):
            return jax.ShapeDtypeStruct(shape, jnp.float32)

        with at.disable_typechecking():
            obs = dataclasses.replace(
                obs,
                state=spec((batch_size, self.state_dim)),
                lip_visual=spec((batch_size, self.visual_views, 64, self.visual_dim)) if self.use_lip_visual else None,
                lip_tactile=spec(
                    (
                        batch_size,
                        self.history_frames,
                        self.tactile_sensors * self.tactile_tokens_per_sensor,
                        self.tactile_dim,
                    )
                ),
                lip_proprio=spec((batch_size, self.history_frames, self.state_dim)),
                lip_mask=jax.ShapeDtypeStruct((batch_size, self.lip_token_count), jnp.bool_),
            )
        return obs, target


class LipPrefix(nnx.Module):
    def __init__(self, cfg, *, width, rngs):
        self.cfg = cfg
        self.width = width
        if cfg.use_lip_visual:
            self.visual_norm = nnx.LayerNorm(cfg.visual_dim, rngs=rngs)
            self.visual_down = nnx.Linear(cfg.visual_dim, cfg.visual_adapter_dim, rngs=rngs)
            self.visual_up = nnx.Linear(
                cfg.visual_adapter_dim, cfg.visual_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )
            self.visual_proj = nnx.Linear(cfg.visual_dim, width, rngs=rngs)
        self.tactile_norm = nnx.LayerNorm(cfg.tactile_dim, rngs=rngs)
        self.tactile_proj = nnx.Linear(cfg.tactile_dim, width, rngs=rngs)
        self.proprio_norm = nnx.LayerNorm(cfg.state_dim, rngs=rngs)
        self.proprio_in = nnx.Linear(cfg.state_dim, cfg.proprio_hidden_dim, rngs=rngs)
        self.proprio_temporal = nnx.Conv(
            cfg.proprio_hidden_dim,
            cfg.proprio_hidden_dim,
            kernel_size=(3,),
            padding="SAME",
            feature_group_count=cfg.proprio_hidden_dim,
            rngs=rngs,
        )
        self.proprio_pointwise = nnx.Linear(cfg.proprio_hidden_dim, cfg.proprio_hidden_dim, rngs=rngs)
        self.proprio_out = nnx.Linear(cfg.proprio_hidden_dim, width, rngs=rngs)
        self.output_norm = nnx.LayerNorm(width, rngs=rngs)

        def emb(shape):
            return nnx.Param(jax.random.normal(rngs.params(), shape) * 0.02)

        self.modality = emb((3, width))
        if cfg.use_lip_visual:
            self.view = emb((cfg.visual_views, width))
            self.spatial = emb((64, width))
        self.time = emb((cfg.history_frames, width))
        self.sensor = emb((cfg.tactile_sensors, width))
        self.sensor_position = emb((cfg.tactile_tokens_per_sensor, width))

    def __call__(self, obs):
        cfg = self.cfg
        if any(x is None for x in (obs.lip_tactile, obs.lip_proprio, obs.lip_mask)):
            raise ValueError("LIP tactile, proprio history and masks are required")
        if cfg.use_lip_visual != (obs.lip_visual is not None):
            raise ValueError("LIP visual input does not match use_lip_visual")
        b = obs.state.shape[0]
        expected = (
            (b, cfg.history_frames, cfg.tactile_sensors * cfg.tactile_tokens_per_sensor, cfg.tactile_dim),
            (b, cfg.history_frames, cfg.state_dim),
            (b, cfg.lip_token_count),
        )
        for value, shape in zip(
            (obs.lip_tactile, obs.lip_proprio, obs.lip_mask), expected, strict=True
        ):
            if value.shape != shape:
                raise ValueError(f"LIP condition shape {value.shape} != {shape}")
        parts = []
        if cfg.use_lip_visual:
            if obs.lip_visual.shape != (b, cfg.visual_views, 64, cfg.visual_dim):
                raise ValueError("Invalid LIP visual shape")
            visual = obs.lip_visual + self.visual_up(nnx.gelu(self.visual_down(self.visual_norm(obs.lip_visual))))
            visual = self.visual_proj(visual) + self.view[None, :, None] + self.spatial[None, None] + self.modality[0]
            parts.append(visual.reshape(b, -1, self.width))
        tactile = self.tactile_proj(self.tactile_norm(obs.lip_tactile)).reshape(
            b, cfg.history_frames, cfg.tactile_sensors, cfg.tactile_tokens_per_sensor, self.width
        )
        tactile = tactile + self.time[None, :, None, None] + self.sensor[None, None, :, None]
        tactile = tactile + self.sensor_position[None, None, None] + self.modality[1]
        proprio = self.proprio_in(self.proprio_norm(obs.lip_proprio))
        proprio = proprio + self.proprio_pointwise(nnx.gelu(self.proprio_temporal(proprio)))
        proprio = self.proprio_out(proprio) + self.time[None] + self.modality[2]
        parts.extend((tactile.reshape(b, -1, self.width), proprio))
        tokens = jnp.concatenate(parts, axis=1)
        return self.output_norm(tokens) * obs.lip_mask[..., None]


class LipPi05(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.lip_prefix = LipPrefix(config, width=gemma.get_config(config.paligemma_variant).width, rngs=rngs)

    def embed_prefix(self, obs):
        tokens, valid, ar = super().embed_prefix(obs)
        extra = self.lip_prefix(obs).astype(tokens.dtype)
        language_len = 0 if obs.tokenized_prompt is None else obs.tokenized_prompt.shape[1]
        split = tokens.shape[1] - language_len
        return (
            jnp.concatenate((tokens[:, :split], extra, tokens[:, split:]), axis=1),
            jnp.concatenate((valid[:, :split], obs.lip_mask, valid[:, split:]), axis=1),
            jnp.concatenate((ar[:split], jnp.zeros(extra.shape[1], dtype=jnp.bool_), ar[split:])),
        )

    def compute_loss(self, rng, observation, actions, *, train=False):
        # Cached LIP features use the original pixels. Disable independent geometric/color
        # augmentation of SigLIP images to keep the two branches paired in v1.
        return super().compute_loss(rng, observation, actions, train=False)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, solver="euler"):
        if solver not in ("euler", "heun") or num_steps < 1:
            raise ValueError("Use euler/heun and a positive number of steps")
        obs = base.preprocess_observation(None, observation, train=False)
        if noise is None:
            noise = jax.random.normal(rng, (obs.state.shape[0], self.action_horizon, self.action_dim))
        tokens, valid, ar = self.embed_prefix(obs)
        _, cache = self.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(valid, ar), positions=jnp.cumsum(valid, axis=1) - 1
        )

        def velocity(x, time):
            suffix, mask, suffix_ar, cond = self.embed_suffix(obs, x, jnp.full((x.shape[0],), time))
            full_mask = jnp.concatenate(
                (
                    jnp.broadcast_to(valid[:, None], (x.shape[0], suffix.shape[1], valid.shape[1])),
                    make_attn_mask(mask, suffix_ar),
                ),
                axis=-1,
            )
            positions = jnp.sum(valid, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
            (_, out), _ = self.PaliGemma.llm(
                [None, suffix], mask=full_mask, positions=positions, kv_cache=cache, adarms_cond=[None, cond]
            )
            return self.action_out_proj(out[:, -self.action_horizon :])

        dt = -1.0 / num_steps

        def step(i, x):
            time = 1.0 + i * dt
            first = velocity(x, time)
            proposed = x + dt * first
            if solver == "euler":
                return proposed
            return x + dt * (first + velocity(proposed, time + dt)) * 0.5

        return jax.lax.fori_loop(0, num_steps, step, noise)
