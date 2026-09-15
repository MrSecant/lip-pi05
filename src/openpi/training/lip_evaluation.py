"""Sample the same LIP evaluation windows; decode in the original PyTorch runtime."""

import json
import logging
import os
from pathlib import Path
import subprocess
import time
from contextlib import nullcontext

from flax import nnx
import jax
import numpy as np

from openpi.lip_data import LipCacheDataset
from openpi.lip_sampling import reference_metric_indices
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import TokenizePrompt
from openpi.training import sharding
from openpi.training.lip_logging import steps_per_epoch


def write_run_assets(config):
    ds = LipCacheDataset(config.data.cache_path, "train")
    output = config.checkpoint_dir / "lip_data_contract"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != ds.manifest:
        raise ValueError("Cannot resume with a different LIP data contract")
    path.write_text(json.dumps(ds.manifest, indent=2))
    np.savez(output / "latent_stats.npz", mean=ds.mean, std=ds.std)
    language = {
        "train_prompts": list(getattr(config.data, "train_prompts", ())),
        "eval_prompt": getattr(config.data, "eval_prompt", None),
        "seed": getattr(config.data, "prompt_seed", 42),
        "sampling": "uniform per sample and reference epoch; deterministic by seed, epoch, pair_index",
    }
    language_path = output / "language.json"
    if language_path.exists() and json.loads(language_path.read_text()) != language:
        raise ValueError("Cannot resume with a different language sampling contract")
    language_path.write_text(json.dumps(language, indent=2))


def metric_indices(count, *, max_batches, reference_batch_size):
    return reference_metric_indices(count, max_batches, reference_batch_size)


def evaluate(model, cache_path, output, *, batch_size=8, allow_partial=False, mesh=None, prompt=None,
             solvers_override=(), sampling_steps_override=()):
    if batch_size < 1:
        raise ValueError("Evaluation batch_size must be positive")
    if mesh is not None and batch_size % mesh.size:
        raise ValueError("Evaluation global batch must be divisible by the device count")
    data_sharding = None if mesh is None else jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(cache_path) / "manifest.json").read_text())
    cfg = manifest["stage2_config"]
    tokenizer = TokenizePrompt(PaligemmaTokenizer(model.max_token_len), discrete_state_input=True)
    evaluation = cfg["evaluation"]["target_metrics"]
    seed = int(evaluation.get("seed", 2026))
    prediction_manifest = {
        "cache_path": str(cache_path),
        "rows_sha256": manifest["rows_sha256"],
        "stage1_sha256": manifest["stage1_sha256"],
        "allow_partial": allow_partial,
        "groups": [],
        "flow_losses": {},
        "prompt": prompt if prompt is not None else manifest.get("prompt"),
    }
    model.eval()
    graph, params = nnx.split(model)
    def sample_losses(weights, obs, target, key):
        losses = nnx.merge(graph, weights).compute_loss(key, obs, target)
        return losses.reshape((target.shape[0], -1)).mean(axis=1)
    loss_fn = jax.jit(sample_losses, out_shardings=data_sharding)
    compiled = {}
    for split in manifest["split_names"]:
        ds = LipCacheDataset(cache_path, split, allow_partial=allow_partial)
        settings = evaluation["train" if split == "train" else "eval"]
        indices = metric_indices(
            len(ds), max_batches=settings.get("max_batches"), reference_batch_size=int(cfg["train"]["batch_size"])
        )
        solvers = solvers_override or settings.get("solvers", [cfg["model"]["sampling"]["solver"]])
        sampling_steps = sampling_steps_override or settings["sampling_steps"]
        if any(s not in ("euler", "heun") for s in solvers) or any(int(n) < 1 for n in sampling_steps):
            raise ValueError("Invalid LIP evaluation solver or step count")
        if not len(indices):
            raise ValueError(f"No evaluation samples for {split}")
        started = time.monotonic()
        logging.info("LIP eval %s: %s samples, solvers=%s, steps=%s", split, len(indices), solvers, sampling_steps)
        pair_ids = np.asarray([int(ds.identity(i)["pair_index"]) for i in indices], dtype=np.int64)
        outputs = {}
        loss_sum = 0.0
        for solver in solvers:
            for steps in sampling_steps:
                job = (solver, int(steps))
                if job not in compiled:
                    compiled[job] = jax.jit(
                        lambda weights, obs, noise, s=solver, n=int(steps): nnx.merge(graph, weights).sample_actions(
                            jax.random.key(seed), obs, noise=noise, num_steps=n, solver=s
                        ),
                        out_shardings=data_sharding,
                    )
                name = f"{split}_{solver}_{steps}"
                outputs[job] = np.lib.format.open_memmap(
                    output / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(indices), 32, 128)
                )
                np.save(output / f"{name}_pairs.npy", pair_ids)
                prediction_manifest["groups"].append(
                    {
                        "split": split,
                        "solver": solver,
                        "steps": steps,
                        "predictions": f"{name}.npy",
                        "pair_ids": f"{name}_pairs.npy",
                        "count": len(indices),
                    }
                )
        # Read and tokenize each batch once, with identical per-pair noise for every sampler.
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            samples = []
            for i in chosen:
                sample = ds[int(i)]
                if prompt is not None:
                    sample = {**sample, "prompt": prompt}
                samples.append(tokenizer(sample))
            valid_count = len(samples)
            samples.extend([samples[-1]] * (batch_size - valid_count))
            batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
            if data_sharding is not None:
                batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
            observation = Observation.from_dict(batch)
            with sharding.set_mesh(mesh) if mesh is not None else nullcontext():
                losses = loss_fn(params, observation, batch["actions"],
                    jax.random.fold_in(jax.random.key(seed), start))
            loss_sum += float(np.asarray(losses)[:valid_count].sum())
            padded_pairs = np.pad(pair_ids[start : start + valid_count],
                (0, batch_size - valid_count), mode="edge")
            noise = jax.vmap(
                lambda p: jax.random.normal(jax.random.fold_in(jax.random.key(seed), p), (32, 128))
            )(padded_pairs)
            if data_sharding is not None:
                noise = jax.device_put(noise, data_sharding)
            for job, predictions in outputs.items():
                with sharding.set_mesh(mesh) if mesh is not None else nullcontext():
                    predicted = compiled[job](params, observation, noise)
                predictions[start : start + valid_count] = np.asarray(predicted)[:valid_count]
            if start == 0 or (start // batch_size + 1) % 25 == 0 or start + valid_count == len(indices):
                logging.info("LIP eval %s: %s/%s samples, all samplers, %.1fs elapsed",
                    split, start + valid_count, len(indices), time.monotonic() - started)
        for predictions in outputs.values():
            predictions.flush()
        prediction_manifest["flow_losses"][split] = loss_sum / len(indices)
    (output / "predictions.json").write_text(json.dumps(prediction_manifest, indent=2))
    return output


def evaluation_due(config, step):
    if getattr(config.data, "cache_path", None) is None:
        return False
    interval = getattr(config, "lip_eval_every_steps", 0)
    if interval > 0:
        return step > 0 and (step % interval == 0 or step == config.num_train_steps)
    if config.lip_eval_every_epochs <= 0:
        return False
    epoch_steps = steps_per_epoch(config.data.cache_path, config.batch_size)
    epoch, remainder = divmod(step, epoch_steps)
    return step == config.num_train_steps or (
        step > 0 and remainder == 0 and (epoch == 1 or epoch % config.lip_eval_every_epochs == 0)
    )


def on_train_step(config, state, step):
    cache_path = getattr(config.data, "cache_path", None)
    if not evaluation_due(config, step):
        return {}
    weights = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, weights)
    output_path = config.checkpoint_dir / "lip_eval" / f"step_{step:08d}"
    attempt = 0
    while output_path.exists():
        attempt += 1
        output_path = config.checkpoint_dir / "lip_eval" / f"step_{step:08d}_attempt_{attempt}"
    output = evaluate(model, cache_path, output_path, batch_size=config.lip_eval_batch_size,
        mesh=sharding.make_mesh(config.fsdp_devices), prompt=getattr(config.data, "eval_prompt", None),
        solvers_override=config.lip_eval_solvers, sampling_steps_override=config.lip_eval_sampling_steps)
    logging.info("LIP eval step %s: sampling complete; starting decoded metrics in %s", step, output)
    script = Path(__file__).resolve().parents[3] / "scripts/evaluate_lip_predictions.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(script.parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [config.lip_decoder_python, str(script), "--predictions", str(output), "--device", config.lip_decoder_device],
        check=True,
        env=env,
    )
    results = json.loads((output / "metrics.json").read_text())
    results["evaluated_weights"] = "ema" if state.ema_params is not None else "raw"
    results["training_step"] = step
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    return results["scalars"]
