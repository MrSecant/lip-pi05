"""NumPy/Zarr boundary shared by the offline PyTorch exporter and JAX loader."""

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

ROW_FIELDS = (
    "pair_index",
    "split_id",
    "dataset_index",
    "source_id",
    "source_idx",
    "episode_index",
    "anchor_t",
    "action_start",
    "action_end",
    "base_mode",
    "remove_index",
)
ROW_DTYPE = np.dtype([(key, "<i8") for key in ROW_FIELDS])
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_alignment(rows, latent_root):
    if len(np.unique(rows["pair_index"])) != len(rows):
        raise ValueError("Duplicate pair_index in manifest")
    meta = latent_root["meta"]
    pair = rows["pair_index"]
    if np.any(pair < 0) or np.any(pair >= latent_root["data/interaction_latent"].shape[0]):
        raise ValueError("pair_index outside latent cache")
    for key in (
        "split_id",
        "dataset_index",
        "source_idx",
        "episode_index",
        "anchor_t",
        "action_start",
        "action_end",
        "base_mode",
    ):
        if not np.array_equal(rows[key], np.asarray(meta[key][:])[pair]):
            raise ValueError(f"Manifest/latent mismatch: {key}")
    if np.any(rows["action_start"] != rows["anchor_t"]) or np.any(rows["action_end"] - rows["action_start"] != 128):
        raise ValueError("Expected the original anchor-aligned 128-frame target")
    if np.any((rows["base_mode"] == 1) != (rows["remove_index"] >= 0)):
        raise ValueError("Missing or unexpected remove-hand frame index")


class LipCacheDataset:
    """Read original RGB and frozen conditions; do not re-window or re-fit statistics."""

    def __init__(self, root, split="train", *, allow_partial=False):
        self.path = Path(root)
        self.split = split
        self.allow_partial = allow_partial
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if not self.manifest["complete"]:
            raise ValueError("Condition export is incomplete")
        if not self.manifest["full_coverage"] and not allow_partial:
            raise ValueError("Diagnostic subset cannot be used for training; export full coverage first")
        self.rows = np.load(self.path / "rows.npy", mmap_mode="r")
        if sha256_file(self.path / "rows.npy") != self.manifest["rows_sha256"]:
            raise ValueError("Manifest rows checksum mismatch")
        self.latent = zarr.open_group(self.manifest["latent_cache_path"], mode="r")
        if not self.latent.attrs.get("complete", False):
            raise ValueError("Incomplete source latent cache")
        if self.latent.attrs["stage1_checkpoint_sha256"] != self.manifest["stage1_sha256"]:
            raise ValueError("Stage1 checkpoint identity mismatch")
        validate_alignment(self.rows, self.latent)
        self.cached_rows = np.load(self.path / "cached_rows.npy")
        if sha256_file(self.path / "cached_rows.npy") != self.manifest["cached_rows_sha256"]:
            raise ValueError("Feature row mapping checksum mismatch")
        if len(np.unique(self.cached_rows)) != len(self.cached_rows):
            raise ValueError("Duplicate cached row")
        if np.any(self.cached_rows < 0) or np.any(self.cached_rows >= len(self.rows)):
            raise ValueError("Cached row outside manifest")
        if self.manifest["full_coverage"] and not np.array_equal(self.cached_rows, np.arange(len(self.rows))):
            raise ValueError("Full coverage claimed but rows are missing or reordered")
        split_id = self.manifest["split_names"].index(split)
        self.indices = np.flatnonzero(self.rows["split_id"][self.cached_rows] == split_id)
        self.visual = np.load(self.path / "visual.npy", mmap_mode="r")
        self.tactile = np.load(self.path / "tactile.npy", mmap_mode="r")
        self.proprio = np.load(self.path / "proprio.npy", mmap_mode="r")
        count = len(self.cached_rows)
        for array, shape in (
            (self.visual, (count, 3, 64, 256)),
            (self.tactile, (count, 8, 8, 64)),
            (self.proprio, (count, 8, self.manifest["state_dim"])),
        ):
            if array.shape != shape:
                raise ValueError(f"Feature shape {array.shape} != {shape}")
        stats = self.latent["stats"]
        if tuple(stats.attrs["split_names"]) != ("train",):
            raise ValueError("Latent statistics must be train-only")
        self.mean = np.asarray(stats["latent_mean"][:], dtype=np.float32)
        self.std = np.maximum(np.asarray(stats["latent_std"][:], dtype=np.float32), self.manifest["latent_eps"])
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Non-finite latent statistics")
        self._sources = {}
        self._validated_signatures = self._snapshot_signatures()

    def _snapshot_signatures(self):
        paths = [self.path / name for name in
                 ("manifest.json", "rows.npy", "cached_rows.npy", "visual.npy", "tactile.npy", "proprio.npy")]
        latent_path = Path(self.manifest["latent_cache_path"])
        paths.extend(path for path in (latent_path / "zarr.json", latent_path / ".zattrs") if path.exists())
        return {str(path): self._file_signature(path) for path in paths}

    def __getstate__(self):
        # Full validation already ran in the parent. Workers reopen immutable files.
        if self._snapshot_signatures() != self._validated_signatures:
            raise ValueError("Validated cache changed before spawning workers")
        return {"root": str(self.path), "split": self.split, "allow_partial": self.allow_partial,
                "validated_manifest": self.manifest, "signatures": self._validated_signatures,
                "mean": self.mean, "std": self.std}

    @staticmethod
    def _file_signature(path):
        stat = Path(path).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def __setstate__(self, state):
        if "validated_manifest" not in state:
            self.__init__(state["root"], state["split"], allow_partial=state["allow_partial"])
            return
        for path, signature in state["signatures"].items():
            if self._file_signature(path) != signature:
                raise ValueError(f"Validated cache changed before worker startup: {path}")
        self.path, self.split = Path(state["root"]), state["split"]
        self.allow_partial = state["allow_partial"]
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest != state["validated_manifest"]:
            raise ValueError("Validated manifest changed before worker startup")
        self.rows = np.load(self.path / "rows.npy", mmap_mode="r")
        self.cached_rows = np.load(self.path / "cached_rows.npy", mmap_mode="r")
        split_id = self.manifest["split_names"].index(self.split)
        self.indices = np.flatnonzero(self.rows["split_id"][self.cached_rows] == split_id)
        for name in ("visual", "tactile", "proprio"):
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r"))
        self.latent = zarr.open_group(self.manifest["latent_cache_path"], mode="r")
        if not self.latent.attrs.get("complete", False):
            raise ValueError("Incomplete latent cache in worker")
        if self.latent.attrs["stage1_checkpoint_sha256"] != self.manifest["stage1_sha256"]:
            raise ValueError("Stage1 identity changed before worker startup")
        self.mean, self.std = state["mean"], state["std"]
        self._sources = {}
        self._validated_signatures = state["signatures"]
        if self._snapshot_signatures() != self._validated_signatures:
            raise ValueError("Validated cache changed while worker reopened files")

    def raw_actions(self, index):
        """Same 128-frame target interval for a future raw-action Pi0.5 baseline."""
        row = self.identity(index)
        source = zarr.open_group(self.manifest["sources"][int(row["source_id"])]["zarr_path"], mode="r")
        return np.asarray(
            source["data/action_30hz"][int(row["action_start"]) : int(row["action_end"]), :14], dtype=np.float32
        )

    def __len__(self):
        return len(self.indices)

    def identity(self, index):
        return self.rows[self.cached_rows[self.indices[index]]]

    def images(self, row):
        source_id = int(row["source_id"])
        if source_id not in self._sources:
            source = self.manifest["sources"][source_id]
            self._sources[source_id] = zarr.open_group(source["zarr_path"], mode="r")
        source = self._sources[source_id]
        pixels = np.asarray(source["data/camera"][int(row["anchor_t"])]).copy()
        if row["base_mode"] == 1:
            pixels[..., :3] = source["data/camera_base_remove_hand"][int(row["remove_index"])]
        if pixels.dtype != np.uint8 or pixels.shape != (224, 224, 9):
            raise ValueError(f"Unexpected RGB layout: {pixels.shape}, {pixels.dtype}")
        return {key: np.ascontiguousarray(pixels[..., i * 3 : (i + 1) * 3]) for i, key in enumerate(IMAGE_KEYS)}

    def __getitem__(self, index):
        feature_row = int(self.indices[index])
        row = self.identity(index)
        target = np.asarray(self.latent["data/interaction_latent"][int(row["pair_index"])], dtype=np.float32)
        if self.manifest["normalize_latent"]:
            target = (target - self.mean) / self.std
        proprio = np.asarray(self.proprio[feature_row], dtype=np.float32)
        return {
            "image": self.images(row),
            "image_mask": {key: np.bool_(True) for key in IMAGE_KEYS},
            "state": proprio[-1].copy(),
            "lip_visual": np.asarray(self.visual[feature_row], dtype=np.float32),
            "lip_tactile": np.asarray(self.tactile[feature_row], dtype=np.float32),
            "lip_proprio": proprio,
            "lip_mask": np.ones(264, dtype=bool),
            "prompt": self.manifest["prompt"],
            "actions": target,
        }
