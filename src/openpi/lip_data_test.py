import json
import os
import pickle

import numpy as np
import pytest
import zarr

from openpi.lip_data import ROW_DTYPE
from openpi.lip_data import LipCacheDataset
from openpi.lip_data import sha256_file
from openpi.lip_data import validate_alignment


def array(group, name, value):
    value = np.asarray(value)
    result = group.create_dataset(name, shape=value.shape, dtype=value.dtype)
    result[:] = value
    return result


@pytest.fixture
def cache(tmp_path):
    latent_path = tmp_path / "latent.zarr"
    latent = zarr.open_group(str(latent_path), mode="w")
    latent.attrs.update(complete=True, stage1_checkpoint_sha256="test")
    array(latent.require_group("data"), "interaction_latent", np.full((2, 32, 128), 5, np.float16))
    rows = np.zeros(2, dtype=ROW_DTYPE)
    rows["pair_index"] = [0, 1]
    rows["dataset_index"] = [0, 1]
    rows["anchor_t"] = [0, 1]
    rows["action_start"] = [0, 1]
    rows["action_end"] = [128, 129]
    rows["base_mode"] = [0, 1]
    rows["remove_index"] = [-1, 0]
    meta = latent.require_group("meta")
    for key in rows.dtype.names:
        array(meta, key, rows[key])
    stats = latent.require_group("stats")
    stats.attrs["split_names"] = ["train"]
    array(stats, "latent_mean", np.full(128, 1, np.float32))
    array(stats, "latent_std", np.full(128, 2, np.float32))
    source_path = tmp_path / "replay.zarr"
    source = zarr.open_group(str(source_path), mode="w").require_group("data")
    array(source, "camera", np.full((2, 224, 224, 9), 10, np.uint8))
    array(source, "camera_base_remove_hand", np.full((1, 224, 224, 3), 77, np.uint8))
    array(source, "action_30hz", np.arange(129 * 62, dtype=np.float32).reshape(129, 62))
    root = tmp_path / "cache"
    root.mkdir()
    np.save(root / "rows.npy", rows)
    np.save(root / "cached_rows.npy", np.arange(2))
    np.save(root / "visual.npy", np.zeros((2, 3, 64, 256), np.float16))
    np.save(root / "tactile.npy", np.zeros((2, 8, 8, 64), np.float16))
    np.save(root / "proprio.npy", np.ones((2, 8, 14), np.float32))
    manifest = dict(
        complete=True,
        full_coverage=True,
        latent_cache_path=str(latent_path),
        stage1_sha256="test",
        rows_sha256=sha256_file(root / "rows.npy"),
        cached_rows_sha256=sha256_file(root / "cached_rows.npy"),
        split_names=["train"],
        sources=[{"zarr_path": str(source_path)}],
        latent_eps=1e-6,
        state_dim=14,
        normalize_latent=True,
        prompt="Peel the cucumber.",
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, latent, rows


def test_target_and_remove_hand_rgb(cache):
    root, _, _ = cache
    ds = LipCacheDataset(root)
    original, removed = ds[0], ds[1]
    np.testing.assert_array_equal(original["actions"], 2)
    np.testing.assert_array_equal(original["image"]["base_0_rgb"], 10)
    np.testing.assert_array_equal(removed["image"]["base_0_rgb"], 77)
    np.testing.assert_array_equal(removed["image"]["left_wrist_0_rgb"], 10)
    np.testing.assert_array_equal(removed["state"], removed["lip_proprio"][-1])


def test_alignment_detects_wrong_anchor(cache):
    _, latent, rows = cache
    rows["anchor_t"][0] = 99
    with pytest.raises(ValueError, match="anchor_t"):
        validate_alignment(rows, latent)


def test_partial_cache_rejected_for_training(cache):
    root, _, _ = cache
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["full_coverage"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Diagnostic subset"):
        LipCacheDataset(root)


def test_spawn_serialization_reopens_files(cache):
    root, _, _ = cache
    data = pickle.dumps(LipCacheDataset(root))
    assert len(data) < 4096
    restored = pickle.loads(data)
    np.testing.assert_array_equal(restored[1]["actions"], 2)


def test_raw_action_baseline_keeps_128_frames(cache):
    root, _, _ = cache
    result = LipCacheDataset(root).raw_actions(1)
    assert result.shape == (128, 14)
    np.testing.assert_array_equal(result[0], np.arange(62, 76, dtype=np.float32))


def test_worker_reopen_skips_duplicate_full_scan(cache, monkeypatch):
    root, _, _ = cache
    parent = LipCacheDataset(root)
    serialized = pickle.dumps(parent)
    def unexpected_scan(*args):
        raise AssertionError("Worker repeated full metadata scan")
    monkeypatch.setattr("openpi.lip_data.validate_alignment", unexpected_scan)
    worker = pickle.loads(serialized)
    np.testing.assert_array_equal(worker[1]["actions"], parent[1]["actions"])
    np.testing.assert_array_equal(worker[1]["image"]["base_0_rgb"], parent[1]["image"]["base_0_rgb"])
    assert isinstance(worker.visual, np.memmap)


def test_worker_rejects_changed_file_after_serialization(cache):
    root, _, _ = cache
    serialized = pickle.dumps(LipCacheDataset(root))
    path = root / "visual.npy"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed before worker startup"):
        pickle.loads(serialized)


def test_parent_rejects_changed_file_before_serialization(cache):
    root, _, _ = cache
    parent = LipCacheDataset(root)
    path = root / "tactile.npy"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="changed before spawning"):
        pickle.dumps(parent)
