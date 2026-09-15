"""Use LIP's existing metric functions and decoder on aligned JAX predictions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from openpi.lip_data import sha256_file
from openpi.training.lip_logging import sampling_metric_key


@torch.no_grad()
def decode_metrics(model, latent, action_target, tactile_target, normalizer, force_normalizer):
    from LIP.train_stage_two import _decoded_l1_sums
    from LIP.train_stage_two import _pack_decode_target_tactile

    tactile_target = _pack_decode_target_tactile(tactile_target).to(latent.device)
    action_target = action_target.to(latent.device)
    out = model.decoder(latent, target_horizon=128, tactile_action=None, return_dict=True)
    values = _decoded_l1_sums(
        out.action, action_target, out.tactile_mean, tactile_target, normalizer, segment_frames=32
    )
    scale = normalizer.tactile.scale.to(latent.device).view(1, 1, 12, 1, 1)
    offset = normalizer.tactile.offset.to(latent.device).view(1, 1, 12, 1, 1)

    def resultant(tactile):
        raw = (tactile.float() - offset) / scale
        return raw.reshape(raw.shape[0], 128, 4, 3, 35, 20)[:, :, :, :2].sum(dim=(-2, -1))

    target_force = resultant(tactile_target)
    predictions = {"map_force": resultant(out.tactile_mean)}
    if model.force_decoder is not None:
        predictions["direct_force"] = force_normalizer.unnormalize_tensor(
            model.force_decoder(latent, target_horizon=128)
        )
    for name, pred in predictions.items():
        vector = (pred - target_force).abs()
        magnitude = (pred.norm(dim=-1) - target_force.norm(dim=-1)).abs()
        values[f"{name}_vector_l1_unnormalized"] = float(vector.flatten(1).mean(1).sum())
        values[f"{name}_magnitude_l1_unnormalized"] = float(magnitude.flatten(1).mean(1).sum())
        for sensor in range(4):
            values[f"{name}/sensor_{sensor}/magnitude_l1_unnormalized"] = float(magnitude[:, :, sensor].mean(1).sum())
    return values


def main(args):
    sys.path.insert(0, str(Path(args.lip_root).resolve().parent))
    from LIP.data.stage2_dataset import build_stage2_datasets_from_config
    from LIP.encode_stage1_latents import build_stage1_model
    from LIP.train_stage_two import _target_batch_metrics
    from LIP.train_stage_two import _target_segment_batch_metrics

    path = Path(args.predictions)
    prediction_manifest = json.loads((path / "predictions.json").read_text())
    cache_path = Path(prediction_manifest["cache_path"])
    manifest = json.loads((cache_path / "manifest.json").read_text())
    if prediction_manifest["rows_sha256"] != sha256_file(cache_path / "rows.npy"):
        raise ValueError("Predictions use a different sample manifest")
    if prediction_manifest["stage1_sha256"] != manifest["stage1_sha256"]:
        raise ValueError("Prediction/decoder Stage1 mismatch")
    cfg = manifest["stage2_config"]
    if sha256_file(cfg["stage1"]["checkpoint_path"]) != manifest["stage1_sha256"]:
        raise ValueError("Stage1 checkpoint changed")
    train, evaluation, state = build_stage2_datasets_from_config(cfg)
    datasets = {"train": train, **evaluation}
    device = torch.device(args.device)
    model = build_stage1_model(state["config"], train.condition_dataset, device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    scalars = {}
    for name, value in prediction_manifest.get("flow_losses", {}).items():
        key = "train/flow_loss_eval_subset" if name == "train" else f"eval/{name}/loss"
        scalars[key] = value
    for group in prediction_manifest["groups"]:
        started = time.monotonic()
        print(f"Decoding {group['split']} {group['solver']} {group['steps']}: {group['count']} samples", flush=True)
        ds = datasets[group["split"]]
        predictions = np.load(path / group["predictions"], mmap_mode="r")
        pair_ids = np.load(path / group["pair_ids"])
        if predictions.shape != (len(pair_ids), 32, 128) or len(pair_ids) != len(np.unique(pair_ids)):
            raise ValueError("Invalid or duplicated prediction rows")
        local = np.searchsorted(ds.pair_rows, pair_ids)
        if np.any(local >= len(ds)) or not np.array_equal(ds.pair_rows[local], pair_ids):
            raise ValueError("Prediction rows are not in the requested split")
        totals, decoded, oracle = defaultdict(float), defaultdict(float), defaultdict(float)
        for start in range(0, len(pair_ids), args.batch_size):
            chosen = local[start : start + args.batch_size]
            items = [ds._get_item(int(i), include_decode_targets=True) for i in chosen]
            target = torch.stack([x["target"] for x in items]).to(device)
            pred = torch.tensor(np.asarray(predictions[start : start + len(chosen)]), device=device)
            if not torch.isfinite(pred).all():
                raise ValueError("Non-finite predictions")
            raw_target, raw_pred = ds.denormalize_target(target), ds.denormalize_target(pred)
            for p, t, prefix in ((pred, target, "normalized_"), (raw_pred, raw_target, "")):
                l1, cosine, n = _target_batch_metrics(p, t)
                totals[prefix + "l1"] += l1
                totals[prefix + "cosine_similarity"] += cosine
                totals[prefix + "mse"] += float((p - t).square().flatten(1).mean(1).sum())
                for segment, (l1, cosine, _) in _target_segment_batch_metrics(
                    p, t, target_segment_length=8, segment_frames=32
                ).items():
                    totals[f"{segment}/{prefix}l1"] += l1
                    totals[f"{segment}/{prefix}cosine_similarity"] += cosine
            metric_args = dict(
                action_target=torch.stack([x["decode_target_action"] for x in items]),
                tactile_target=torch.stack([x["decode_target_tactile"] for x in items]),
                normalizer=ds.condition_dataset.base_dataset.normalizer,
                force_normalizer=ds.condition_dataset.base_dataset.force_normalizer,
            )
            sampled = decode_metrics(model, raw_pred, **metric_args)
            reference = decode_metrics(model, raw_target, **metric_args)
            for key, value in sampled.items():
                decoded[key] += value
            for key, value in reference.items():
                oracle[key] += value
            if start == 0 or (start // args.batch_size + 1) % 128 == 0 or start + len(chosen) == len(pair_ids):
                print(f"Decode {group['split']} {group['solver']} {group['steps']}: "
                      f"{start + len(chosen)}/{len(pair_ids)}, {time.monotonic() - started:.1f}s", flush=True)
        split_key = "train" if group["split"] == "train" else f"eval/{group['split']}"
        solvers = [item["solver"] for item in prediction_manifest["groups"] if item["split"] == group["split"]]
        job_key = sampling_metric_key(group["solver"], group["steps"], solvers)
        for namespace, values in (("latent_metrics", totals), ("decoded_metrics", decoded)):
            for key, value in values.items():
                if key != "count":
                    scalars[f"{namespace}/{split_key}/{job_key}/{key}"] = value / len(pair_ids)
        for key, value in oracle.items():
            if key != "count":
                scalars[f"decoded_oracle/{split_key}/{key}"] = value / len(pair_ids)
    result = {
        "scalars": scalars,
        "prediction_manifest": prediction_manifest,
        "metric_implementation": "LIP.train_stage_two helpers; added latent MSE; float32 decoder",
        "force_metrics": "Additional signed-XY spatial-sum resultant and direct-force metrics in original tactile units.",
    }
    (path / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--lip-root", default="../LIP")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    main(parser.parse_args())
