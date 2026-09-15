"""Export frozen LIP conditions, preserving the complete Stage2 sample manifest.

Run with the existing LIP PyTorch environment; JAX is not imported here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from openpi.lip_data import ROW_DTYPE
from openpi.lip_data import sha256_file
from openpi.lip_data import validate_alignment


def build_manifest(datasets):
    rows, sources, dataset_lookup = [], [], []
    names = list(datasets)
    for split_id, (name, ds) in enumerate(datasets.items()):
        condition = ds.condition_dataset
        children = getattr(condition, "datasets", (condition,))
        offset = 0
        for source_idx, child in enumerate(children):
            base = child.base_dataset
            source_id = len(sources)
            sources.append(
                {
                    "zarr_path": base.zarr_path,
                    "split": name,
                    "source_idx": source_idx,
                    "camera_views": list(base.camera_views),
                }
            )
            if tuple(base.camera_views) != ("base_0", "left_wrist_0", "right_wrist_0"):
                raise ValueError("Expected original three-view ordering")
            for child_index, window in enumerate(child.windows):
                index = offset + child_index
                remove_index = (
                    base.main_t_to_compact_t(window.anchor_t, window.ep_idx) if window.base_mode == "remove" else -1
                )
                rows.append(
                    (
                        ds.pair_rows[index],
                        split_id,
                        index,
                        source_id,
                        source_idx,
                        window.ep_idx,
                        window.anchor_t,
                        window.action_start,
                        window.action_end,
                        int(window.base_mode == "remove"),
                        remove_index,
                    )
                )
            offset += len(child)
        dataset_lookup.append(ds)
    return np.asarray(rows, dtype=ROW_DTYPE), sources, names, dataset_lookup


@torch.no_grad()
def frozen_features(encoder, patches, tactile):
    visual = encoder.visual_patch_encoder(patches)
    views = torch.arange(visual.shape[1], device=visual.device)
    visual = encoder.visual_norm(visual + encoder.visual_view_emb(views)[None, :, None])
    b, v, _, d = visual.shape
    visual = visual.reshape(b, v, 8, 2, 8, 2, d).mean(dim=(3, 5))
    return visual.reshape(b, v, 64, d), encoder.tactile_encoder(tactile)


class FrozenConditionModule(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, patches, tactile):
        return frozen_features(self.encoder, patches, tactile)


def read_conditions(dataset, index):
    """Read the original histories without fetching unused future decode targets."""
    condition = dataset.condition_dataset
    if hasattr(condition, "resolve_index"):
        _, local, child = condition.resolve_index(int(index))
    else:
        local, child = int(index), condition
    window = child.windows[local]
    start, stop, pad = child._history_range(window, child.base_dataset.window_size)
    proprio = child._left_pad_first(child.base_dataset.get_state(start, stop), pad)
    proprio = child.base_dataset.normalizer.normalize_state_np(proprio)
    visual_indices, _ = child._visual_indices(window)
    patches = child._camera_latent(visual_indices, window)
    return {
        "visual_patches": torch.from_numpy(patches.astype(np.float32, copy=False)),
        "tactile_history": dataset._tactile_history(int(index)),
        "proprio": torch.from_numpy(proprio.astype(np.float32, copy=False)),
    }


def cache_local_visual_frames(base):
    """Reuse immutable per-frame arrays across overlapping condition windows."""
    original = base._load_local_feature_rows
    roots = {}

    @lru_cache(maxsize=1024)
    def one(root, frame):
        return original(np.asarray([frame], dtype=np.int64), **roots[root])[0]

    def cached(indices, **kwargs):
        root = kwargs["local_root"]
        roots.setdefault(root, kwargs)
        return np.ascontiguousarray(np.stack([one(root, int(i)) for i in np.asarray(indices).reshape(-1)]))

    base._load_local_feature_rows = cached


def main(args):
    if args.batch_size < 1 or args.samples_per_group < 0 or args.workers < 1:
        raise ValueError("batch_size/workers must be positive; samples_per_group must be nonnegative")
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(args.lip_root).resolve().parent))
    from LIP.data.stage2_dataset import build_stage2_datasets_from_config
    from LIP.encode_stage1_latents import build_stage1_model

    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    previous = json.loads((output / "manifest.json").read_text()) if args.resume else None
    if previous and previous["complete"]:
        raise ValueError("Refusing to resume a complete cache")
    cfg = yaml.safe_load(Path(args.stage2_config).read_text())
    train, evaluation, state = build_stage2_datasets_from_config(cfg)
    if train.proprio_dim != 14 or train.target_length != 32 or train.target_dim != 128:
        raise ValueError("v1 expects 14 state dimensions and a 32x128 latent")
    stage1_hash = sha256_file(cfg["stage1"]["checkpoint_path"])
    if stage1_hash != train.root.attrs["stage1_checkpoint_sha256"]:
        raise ValueError("Latent cache was encoded by a different Stage1 checkpoint")
    datasets = {"train": train, **evaluation}
    rows, sources, names, lookup = build_manifest(datasets)
    validate_alignment(rows, train.root)
    for dataset in datasets.values():
        condition = dataset.condition_dataset
        for child in getattr(condition, "datasets", (condition,)):
            cache_local_visual_frames(child.base_dataset)
    # Check every source/mode, including both dataset boundaries, before full export.
    checked_inputs = 0
    for sid in np.unique(rows["source_id"]):
        for mode in np.unique(rows["base_mode"][rows["source_id"] == sid]):
            group = np.flatnonzero((rows["source_id"] == sid) & (rows["base_mode"] == mode))
            for i in np.unique(group[[0, len(group) // 2, -1]]):
                ds = lookup[int(rows[i]["split_id"])]
                index = int(rows[i]["dataset_index"])
                reference, fast = ds[index], read_conditions(ds, index)
                for key in fast:
                    torch.testing.assert_close(fast[key], reference[key], rtol=0, atol=0)
                checked_inputs += 1
    if previous:
        np.testing.assert_array_equal(rows, np.load(output / "rows.npy"))
    else:
        output.mkdir(parents=True)
        np.save(output / "rows.npy", rows, allow_pickle=False)
    selected = np.arange(len(rows))
    if args.samples_per_group:
        selected = []
        for sid in np.unique(rows["source_id"]):
            for mode in np.unique(rows["base_mode"][rows["source_id"] == sid]):
                group = np.flatnonzero((rows["source_id"] == sid) & (rows["base_mode"] == mode))
                selected.extend(
                    group[
                        np.unique(
                            np.linspace(0, len(group) - 1, min(args.samples_per_group, len(group)), dtype=np.int64)
                        )
                    ].tolist()
                )
        selected = np.asarray(sorted(selected), dtype=np.int64)
    if previous:
        np.testing.assert_array_equal(selected, np.load(output / "cached_rows.npy"))
    else:
        np.save(output / "cached_rows.npy", selected, allow_pickle=False)
    manifest = dict(
        version=1,
        complete=False,
        full_coverage=len(selected) == len(rows),
        stage2_config=cfg,
        stage1_sha256=stage1_hash,
        latent_cache_path=str(train.latent_cache_path),
        rows_sha256=sha256_file(output / "rows.npy"),
        cached_rows_sha256=sha256_file(output / "cached_rows.npy"),
        split_names=names,
        split_counts={name: len(ds) for name, ds in datasets.items()},
        sources=sources,
        state_dim=train.proprio_dim,
        history_frames=8,
        visual_offsets=[-12, -8, -4, 0],
        prompt=args.prompt,
        normalize_latent=train.normalize_latent,
        latent_eps=train.latent_normalization_eps,
        feature_count=len(selected),
        feature_dtype="float32" if args.float32 else "float16",
        fast_condition_exact_checks=checked_inputs,
        sample_identity="split/source/episode/anchor/base_mode/chunk128",
        rgb_policy="anchor_t; base_0 replaced by matching remove-hand image iff base_mode=remove",
        augmentation="none; preserve paired cached/raw views",
        encoding="float32 eval mode; frozen visual norm/view embedding; spatial 2x2 mean; tactile window t-7:t",
    )
    if previous:
        for key in manifest:
            if key != "fast_condition_exact_checks" and previous.get(key) != manifest[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "stage2_config.yaml").write_text(yaml.safe_dump(cfg))
    dtype = np.float32 if args.float32 else np.float16

    def feature_array(name, shape, array_dtype):
        if previous:
            array = np.load(output / name, mmap_mode="r+")
            if array.shape != shape or array.dtype != array_dtype:
                raise ValueError(f"Resume array mismatch: {name}")
            return array
        return np.lib.format.open_memmap(output / name, mode="w+", dtype=array_dtype, shape=shape)

    visual = feature_array("visual.npy", (len(selected), 3, 64, 256), dtype)
    tactile = feature_array("tactile.npy", (len(selected), 8, 8, 64), dtype)
    proprio = feature_array("proprio.npy", (len(selected), 8, 14), np.float32)
    resume_start = int(json.loads((output / "progress.json").read_text())["completed"]) if previous else 0
    if not 0 <= resume_start <= len(selected):
        raise ValueError("Invalid resume progress")
    device = torch.device(args.device)
    model = build_stage1_model(state["config"], train.condition_dataset, device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    features = FrozenConditionModule(model.encoder)
    if args.devices:
        device_ids = [int(x) for x in args.devices.split(",")]
        if len(set(device_ids)) != len(device_ids) or device.index != device_ids[0]:
            raise ValueError("Unique --devices must start with --device index")
        features = torch.nn.DataParallel(features, device_ids=device_ids)

    def load_row(i):
        return read_conditions(lookup[int(rows[i]["split_id"])], int(rows[i]["dataset_index"]))

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as readers:
        for start in range(resume_start, len(selected), args.batch_size):
            ids = selected[start : start + args.batch_size]
            items = list(readers.map(load_row, ids))
            patches = torch.stack([x["visual_patches"] for x in items]).to(device)
            touch = torch.stack([x["tactile_history"] for x in items]).to(device)
            v, t = features(patches, touch)
            p = torch.stack([x["proprio"] for x in items])
            if not torch.isfinite(v).all() or not torch.isfinite(t).all() or not torch.isfinite(p).all():
                raise ValueError("Non-finite frozen features")
            visual[start : start + len(ids)] = v.cpu().numpy()
            tactile[start : start + len(ids)] = t.cpu().numpy()
            proprio[start : start + len(ids)] = p.numpy()
            done = start + len(ids)
            if start == resume_start or (start - resume_start) // args.batch_size % 20 == 0 or done == len(selected):
                for array in (visual, tactile, proprio):
                    array.flush()
                elapsed = time.monotonic() - started
                progress = dict(
                    completed=done,
                    total=len(selected),
                    seconds=elapsed,
                    resumed_at=resume_start,
                    samples_per_second=(done - resume_start) / elapsed,
                )
                temporary = output / "progress.tmp"
                temporary.write_text(json.dumps(progress, indent=2))
                temporary.replace(output / "progress.json")
                print(json.dumps(progress), flush=True)
    for array in (visual, tactile, proprio):
        array.flush()
    manifest["complete"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {k: manifest[k] for k in ("split_counts", "feature_count", "full_coverage", "rows_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lip-root", default="../LIP")
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--samples-per-group", type=int, default=0, help="Diagnostic subset only; 0 exports every sample"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", help="Optional comma-separated CUDA device IDs for parallel encoding")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete cache with the exact same contract")
    parser.add_argument(
        "--float32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store frozen conditions in FP32 (default); --no-float32 selects FP16 storage",
    )
    main(parser.parse_args())
