"""Inference-only NumPy boundary for the JAX LIP latent policy.

The returned latent is standardized. Stage1 decoding belongs to the caller.
"""

from collections.abc import Mapping
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as base
from openpi.models.lip_pi05 import LipPi05Config
from openpi.models.pi0 import make_attn_mask
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


class _Velocity(nnx.Module):
    def __init__(self, model):
        self.model = model

    def predict(self, observation, latent, time):
        model = self.model
        obs = base.preprocess_observation(None, observation, train=False)
        prefix, valid, ar = model.embed_prefix(obs)
        suffix, suffix_valid, suffix_ar, cond = model.embed_suffix(obs, latent, time)
        valid = jnp.concatenate((valid, suffix_valid), axis=1)
        ar = jnp.concatenate((ar, suffix_ar))
        (_, result), _ = model.PaliGemma.llm(
            [prefix, suffix], mask=make_attn_mask(valid, ar),
            positions=jnp.cumsum(valid, axis=1) - 1, adarms_cond=[None, cond],
        )
        return model.action_out_proj(result[:, -model.action_horizon:])


class LipPi05Policy:
    """Load trained params and sample complete normalized latent chunks.

    Inputs use the training observation dictionary, with batched uint8 RGB and
    FP32 frozen Stage1 features. Supply tokenized text or a local tokenizer file.
    This class does not load training datasets, construct an optimizer, or decode
    robot actions. Configure JAX devices/memory before importing this module.
    """

    def __init__(self, model, config, *, tokenizer=None, jit=True):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.latent_dim = config.action_dim
        self.latent_horizon = config.action_horizon
        self._sample = (nnx_utils.module_jit(model.sample_actions, static_argnames=('num_steps', 'solver'))
                        if jit else model.sample_actions)
        velocity = _Velocity(model)
        self._velocity = nnx_utils.module_jit(velocity.predict) if jit else velocity.predict

    @classmethod
    def from_checkpoint(cls, params_path, *, config=None, tokenizer_path=None, jit=True):
        config = LipPi05Config() if config is None else config
        if isinstance(config, Mapping):
            config = LipPi05Config(**dict(config))
        if not isinstance(config, LipPi05Config):
            raise TypeError('Expected LipPi05Config, not the original robot-action Pi0Config')
        path = Path(params_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        params = base.restore_params(path)
        # Never silently discard a learned LIP projection or prefix parameter.
        model = config.load(params, remove_extra_params=False)
        tokenizer = None
        if tokenizer_path is not None:
            import sentencepiece
            tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
            tokenizer._max_len = config.max_token_len
            tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=Path(tokenizer_path).read_bytes())
        return cls(model, config, tokenizer=tokenizer, jit=jit)

    def _observation(self, inputs):
        cfg = self.config
        if not isinstance(inputs, Mapping):
            raise TypeError('Observation must be a mapping')
        state = np.asarray(inputs['state'])
        if state.ndim != 2 or state.shape[1] != cfg.state_dim or state.shape[0] < 1:
            raise ValueError(f'state must be [B,{cfg.state_dim}]')
        batch = state.shape[0]
        shapes = {
            'state': (batch, cfg.state_dim),
            'lip_tactile': (batch, cfg.history_frames, cfg.tactile_sensors * cfg.tactile_tokens_per_sensor, cfg.tactile_dim),
            'lip_proprio': (batch, cfg.history_frames, cfg.state_dim),
        }
        if cfg.use_lip_visual:
            shapes['lip_visual'] = (batch, cfg.visual_views, 64, cfg.visual_dim)
        elif inputs.get('lip_visual') is not None:
            raise ValueError('No-visual policy must not receive extra Stage1 visual tokens')
        data = {}
        for key, shape in shapes.items():
            value = np.asarray(inputs[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'{key} must be finite with shape {shape}')
            data[key] = value
        if not np.array_equal(data['state'], data['lip_proprio'][:, -1]):
            raise ValueError('Current state and the last proprio history frame must agree')
        images = inputs['image']
        if set(images) != set(IMAGE_KEYS):
            raise ValueError(f'Expected image keys {IMAGE_KEYS}')
        data['image'], data['image_mask'] = {}, {}
        image_masks = inputs.get('image_mask', {key: np.ones(batch, bool) for key in IMAGE_KEYS})
        if set(image_masks) != set(IMAGE_KEYS):
            raise ValueError('Image masks must match all image keys')
        for key in IMAGE_KEYS:
            image = np.asarray(images[key])
            mask = np.asarray(image_masks[key])
            if image.shape != (batch, 224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f'{key} must be uint8 RGB [B,224,224,3]')
            if mask.shape != (batch,) or mask.dtype != np.bool_:
                raise ValueError('Image masks must be boolean [B]')
            data['image'][key], data['image_mask'][key] = image, mask
        mask = np.asarray(inputs.get('lip_mask', np.ones((batch, cfg.lip_token_count), bool)))
        if mask.shape != (batch, cfg.lip_token_count) or mask.dtype != np.bool_:
            raise ValueError('lip_mask has an invalid shape or dtype')
        data['lip_mask'] = mask
        if 'tokenized_prompt' in inputs or 'tokenized_prompt_mask' in inputs:
            tokens = np.asarray(inputs['tokenized_prompt'])
            mask = np.asarray(inputs['tokenized_prompt_mask'])
            if tokens.shape != (batch, cfg.max_token_len) or tokens.dtype.kind not in 'iu':
                raise ValueError('tokenized_prompt has an invalid shape or dtype')
            if mask.shape != tokens.shape or mask.dtype != np.bool_:
                raise ValueError('tokenized_prompt_mask has an invalid shape or dtype')
            if np.any(tokens < 0) or np.any(tokens >= 257152):
                raise ValueError('Prompt token ID outside PaliGemma vocabulary')
        else:
            if self.tokenizer is None:
                raise ValueError('Supply tokenized_prompt/mask or load a local tokenizer_path')
            prompts = inputs.get('prompt')
            prompts = [prompts] * batch if isinstance(prompts, str) else prompts
            if prompts is None or len(prompts) != batch or any(not isinstance(p, str) or not p.strip() for p in prompts):
                raise ValueError('Supply one nonempty prompt per sample')
            pairs = [self.tokenizer.tokenize(p, s) for p, s in zip(prompts, state, strict=True)]
            tokens, mask = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        data['tokenized_prompt'] = tokens.astype(np.int32)
        data['tokenized_prompt_mask'] = mask
        # from_dict performs exactly the training uint8 -> [-1,1] conversion.
        return jax.tree.map(jnp.asarray, base.Observation.from_dict(data))

    def _latent(self, value, batch):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (batch, self.latent_horizon, self.latent_dim) or not np.isfinite(value).all():
            raise ValueError('Latent/noise has an invalid shape or contains non-finite values')
        return jnp.asarray(value)

    def sample(self, observation, *, num_steps=8, solver='euler', seed=0, noise=None):
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1 or solver not in {'euler', 'heun'}:
            raise ValueError('Use a positive integer num_steps and euler/heun')
        obs = self._observation(observation)
        noise = None if noise is None else self._latent(noise, obs.state.shape[0])
        output = self._sample(jax.random.key(seed), obs, num_steps=num_steps, solver=solver, noise=noise)
        result = np.asarray(jax.device_get(output), dtype=np.float32)
        self._latent(result, obs.state.shape[0])
        return result

    def predict_velocity(self, observation, noisy_latent, time):
        """Native OpenPI convention: t=1 is noise, t=0 is target."""
        obs = self._observation(observation)
        latent = self._latent(noisy_latent, obs.state.shape[0])
        time = np.asarray(time, dtype=np.float32)
        if time.shape != (obs.state.shape[0],) or not np.isfinite(time).all() or np.any((time < 0) | (time > 1)):
            raise ValueError('Flow time must be finite [B] in [0,1]')
        value = np.asarray(jax.device_get(self._velocity(obs, latent, jnp.asarray(time))), dtype=np.float32)
        self._latent(value, obs.state.shape[0])
        return value
