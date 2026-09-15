# LIP Pi0.5

Pi0.5 conditioned on frozen LIP Stage1 features, trained to generate joint
action-tactile latents instead of robot-action vectors. This repository retains
OpenPI's source history and license and adds the LIP model, data contract,
evaluation helpers, and a dataset-independent JAX inference entry point.

Upstream base: Physical-Intelligence/openpi, commit
`215abfb217dbac7d5f1273282331b9b1866c0479`.
See [the upstream README](README.openpi.md) for the original project and
[LICENSE](LICENSE) for its Apache-2.0 license. Model weights, tokenizers and
datasets are not distributed here and retain their respective terms.

## Model

- Native Pi0.5 RGB, language and discrete current-state inputs are retained.
- Frozen Stage1 visual features: `[B,3,64,256]`.
- Frozen Stage1 tactile features: `[B,8,8,64]`.
- Normalized proprioceptive history: `[B,8,14]`.
- Trainable LIP prefix: 192 visual + 64 tactile + 8 proprio tokens.
- Denoising target: standardized Stage1 joint latent `[B,32,128]`.
- Matching frozen Stage1 decoder produces 128 robot-action frames.
- The production LIP config uses full-parameter fine-tuning, not LoRA.

The learned visual residual adapter, projections, modality/view/spatial/history
embeddings and proprio temporal encoder are part of the JAX checkpoint. Frozen
Stage1 features are computed offline for training and online by the host LIP
runtime for deployment. Future actions/tactile are not inference inputs.

## Inference Interface

Use a separate inference environment with the dependencies pinned in
`pyproject.toml` and `uv.lock`. Configure visible devices before importing JAX;
do not install or upgrade packages inside an active training environment.

```python
from openpi.lip_inference import LipPi05Policy

policy = LipPi05Policy.from_checkpoint(
    '/models/pi05_lip/55000/params',
    tokenizer_path='/models/paligemma_tokenizer.model',
)
latent_normalized = policy.sample(observation, num_steps=8, solver='euler', seed=42)
# shape [B,32,128], NOT executable robot actions
```

`observation` contains:

| Field | Batched format |
| --- | --- |
| `image` | dict: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`; each uint8 RGB `[B,224,224,3]` |
| `image_mask` | optional matching bool `[B]` masks |
| `state` | normalized float32 `[B,14]`, equal to the final proprio frame |
| `lip_visual` | float32 `[B,3,64,256]` |
| `lip_tactile` | float32 `[B,8,8,64]` |
| `lip_proprio` | normalized float32 `[B,8,14]` |
| `lip_mask` | optional bool `[B,264]`; order visual, tactile, proprio |
| `prompt` | string or list of B strings |

Instead of raw prompts, callers may pass `tokenized_prompt` and
`tokenized_prompt_mask` with shape `[B,max_token_len]` using the native Pi0.5
tokenizer, including its discrete current-state formatting. No tokenizer download
is triggered by this inference loader. Inputs and outputs cross a NumPy boundary
in this initial implementation; GPU zero-copy interoperability is not yet added.

An explicit `noise` array `[B,32,128]` supports reproducible cross-runtime tests.
The default seed is 0: use an explicit changing seed when stochastic predictions
are wanted. Sampling is JAX-native and reuses prefix KV within a complete chunk.
The separate `predict_velocity` API uses native OpenPI time: noise at 1, target at
0. A caller with noise at 0 must reverse time AND negate the velocity.

The caller must denormalize once with the training latent mean/std, decode with
the matching frozen Stage1 model, and unnormalize actions with its normalizer.
For the initial joint-absolute model, the final action is `[B,128,14]` abs_qpos.
Do not substitute a same-shaped Stage1 checkpoint: latent coordinates depend on
its weights. Bundle checkpoint identities, model config, normalization statistics,
camera/sensor/joint order and preprocessing settings with deployment artifacts.

## LIP Integration

The LIP host registers `pi05_lip` under `third_party/adapters/pi05_lip` and pins
this repository under `third_party/vendors/pi05_lip`. The JAX loader strictly
checks all parameter keys and shapes, including the LIP prefix. The host adapter
is inference-only; it does not bridge autograd between frameworks.

This repository does not ship ROS drivers or command a robot. First compare
cached-input inference against the training evaluation path, then validate online
Stage1/DINO feature parity, latency and robot-side safety checks. Unit tests use a
small model; they do not establish real-checkpoint or real-robot equivalence.

## Training

The modified training/evaluation code is retained for reproducibility. Example
config `pi05_lip_cucumber_100k` uses batch 128, 100k steps, evaluation/checkpoints
every 5000 steps and Euler 8/16 evaluation. Configure local data/output locations
before launching. The example cache default is `./data/lip_condition_cache` and
checkpoint default is `./checkpoints/pi05_lip`; no dataset is included.

Condition export uses `scripts/prepare_lip_cache.py --lip-root /path/to/LIP`.
It requires the compatible LIP training dataset/metric implementation. The LIP
`lyc` branch is the inference integration target, not necessarily the complete
dataset/export implementation. Decoded evaluation uses
`scripts/evaluate_lip_predictions.py --lip-root /path/to/compatible/LIP`.

## PyTorch Status

OpenPI includes a PyTorch Pi0.5 implementation and a JAX checkpoint converter,
but neither implements this repository's LIP prefix. The LIP config explicitly
rejects the unmodified PyTorch loader/converter. Do not use a conversion that
silently drops missing/unexpected parameters.

Porting requires equivalent LIP prefix/suffix logic, mapping all new Linear/Conv/
normalization/embedding parameters, and paired tests of prefix features, velocity,
Euler/Heun latent samples and decoded actions with identical inputs and noise.
It should reuse trained weights, not start a new policy training run. Until those
checks pass, JAX is the supported inference backend.
