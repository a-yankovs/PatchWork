#!/usr/bin/env python3
"""
train_tinyvla.py — Fine-tune a TinyVLA / ACT policy on SO101 pick-and-place data.

This script:
  1. Loads all relevant LeRobot datasets from ./data/ and ./skillpatch_data/
     for the full pick-and-place skill (pick_object + place_in_shelf combined).
  2. Merges them into a single training dataset.
  3. Fine-tunes an ACT (Action Chunking with Transformers) policy using LeRobot's
     training infrastructure.
  4. Saves the trained checkpoint to ./checkpoints/tinyvla_pick_and_place/.

Why ACT? It is the policy architecture TinyVLA uses internally and the one
LeRobot ships training code for. It handles the full motion trajectory as
"action chunks" — so a single inference produces a full 50-step motion,
covering the complete pick-and-place sequence without step-by-step fragility.

Usage
-----
# Train on all pick-and-place datasets (auto-detects):
    python scripts/train_tinyvla.py

# Train on specific datasets:
    python scripts/train_tinyvla.py \
        --datasets data/pick_object_v5 data/place_in_shelf data/full_pick_and_place_v1

# Resume from checkpoint:
    python scripts/train_tinyvla.py --resume checkpoints/tinyvla_pick_and_place

# Dry run — print config without training:
    python scripts/train_tinyvla.py --dry-run

# Override training params:
    python scripts/train_tinyvla.py --epochs 200 --batch-size 8 --lr 1e-4

Hardware
--------
Targets AMD ROCm GPU via HIP. Falls back to CPU if no GPU is detected.
For CPU-only debug:
    python scripts/train_tinyvla.py --device cpu --epochs 5 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("train_tinyvla")

# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------

# Datasets that cover the full pick-and-place motion (all phases).
# Ordered by quality / recency — newest first.
FULL_PICK_AND_PLACE_DATASETS = [
    "data/full_pick_and_place_v1",
]

# Component datasets (pick + place separately — also useful for training).
COMPONENT_DATASETS = [
    "data/pick_object_v5",
    "data/place_in_shelf",
    "data/place_in_box_v4",
    "data/place_in_box_v3",
    "data/place_in_box_v2",
    "data/pick_object_v3",
    "data/pick_object_v2",
    "data/pick_object_v1",
]

# Task description injected as the language conditioning input.
# Covers the full pick-and-place motion start to finish.
TASK_DESCRIPTION = (
    "Pick up the object from the table and place it on the shelf."
)

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR    = "checkpoints/tinyvla_pick_and_place"
DEFAULT_EPOCHS        = 20
DEFAULT_BATCH_SIZE    = 16
DEFAULT_LR            = 1e-4
DEFAULT_CHUNK_SIZE    = 20      # ACT action chunk horizon (20 steps @ 50Hz = 0.4 seconds)
DEFAULT_CHUNK_WEIGHT  = 0.1     # ACT temporal ensemble weight
DEFAULT_SEED          = 42


# ---------------------------------------------------------------------------
# Utility: check if a dataset exists and has episodes
# ---------------------------------------------------------------------------

def dataset_exists(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    # LeRobot v2 format: data/chunk-*/file-*.parquet
    has_parquet_v2 = any(p.glob("data/chunk-*/file-*.parquet"))
    # LeRobot v1 format: data/chunk-*/episode_*.parquet
    has_parquet_v1 = any(p.rglob("episode_*.parquet"))
    # meta/info.json is always present in valid datasets
    has_meta = (p / "meta" / "info.json").exists()
    return has_meta or has_parquet_v2 or has_parquet_v1


def count_episodes(path: str) -> int:
    p = Path(path)
    # LeRobot v2: read total_episodes from meta/info.json
    info = p / "meta" / "info.json"
    if info.exists():
        try:
            return json.loads(info.read_text(encoding="utf-8")).get("total_episodes", 0)
        except Exception:
            pass
    # LeRobot v2: count file-*.parquet in data/chunk-*/
    episodes = set()
    for f in p.glob("data/chunk-*/file-*.parquet"):
        try:
            episodes.add(f.stem)
        except ValueError:
            pass
    if episodes:
        return len(episodes)
    # LeRobot v1 fallback: episode_*.parquet
    for f in p.rglob("episode_*.parquet"):
        try:
            episodes.add(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    # meta/episodes.jsonl fallback
    if not episodes:
        meta = p / "meta" / "episodes.jsonl"
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                try:
                    episodes.add(json.loads(line)["episode_index"])
                except Exception:
                    pass
    return len(episodes)


def discover_datasets(root: str = ".") -> list[str]:
    """Find all pick-and-place datasets that exist locally."""
    found = []
    all_candidates = FULL_PICK_AND_PLACE_DATASETS + COMPONENT_DATASETS
    for ds in all_candidates:
        full_path = str(Path(root) / ds) if not Path(ds).is_absolute() else ds
        if dataset_exists(full_path):
            n = count_episodes(full_path)
            if n > 0:
                found.append(full_path)
                logger.info(f"  ✓ {full_path} ({n} episode{'s' if n != 1 else ''})")
            else:
                logger.warning(f"  ⚠ {full_path} exists but has 0 episodes — skipping")
        else:
            logger.debug(f"  — {full_path} not found")
    return found


# ---------------------------------------------------------------------------
# Build LeRobot training config
# ---------------------------------------------------------------------------

def build_lerobot_config(
    datasets: list[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    chunk_size: int,
    device: str,
    seed: int,
    resume: Optional[str],
) -> dict:
    """
    Build a config dict matching LeRobot's train.py / hydra config format.
    Can also be serialised to YAML and passed to lerobot-train.
    """
    # Use dataset root for local-only training (no Hugging Face push)
    repo_ids = [Path(ds).name for ds in datasets]
    roots    = [str(Path(ds).parent) for ds in datasets]

    config = {
        "seed": seed,
        "device": device,
        "output_dir": output_dir,

        # Dataset
        "dataset": {
            "repo_id":     repo_ids[0] if len(repo_ids) == 1 else None,
            "repo_ids":    repo_ids,
            "roots":       roots,
            "push_to_hub": False,
            "video_backend": "pyav",
        },

        # Policy — ACT (Action Chunking with Transformers)
        "policy": {
            "name": "act",
            "chunk_size":             chunk_size,
            "n_action_steps":         chunk_size,
            "n_obs_steps":            1,
            "temporal_ensemble_coeff": DEFAULT_CHUNK_WEIGHT,
            # Vision backbone
            "vision_backbone":  "resnet18",
            "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "replace_final_stride_with_dilation": False,
            # Transformer
            "pre_norm":              False,
            "dim_model":             512,
            "n_heads":               8,
            "dim_feedforward":       3200,
            "feedforward_activation": "relu",
            "n_encoder_layers":      4,
            "n_decoder_layers":      1,
            "use_vae":               True,
            "latent_dim":            32,
            "n_vae_encoder_layers":  4,
            # Loss
            "kl_weight":             10.0,
        },

        # Training loop
        "training": {
            "num_epochs":            epochs,
            "batch_size":            batch_size,
            "lr":                    lr,
            "lr_backbone":           lr * 0.1,       # lower LR for pretrained backbone
            "weight_decay":          1e-4,
            "grad_clip_norm":        10.0,
            "save_checkpoint_every": max(1, epochs // 10),
            "log_every":             10,
            "eval_every":            max(1, epochs // 5),
            "resume_from":           resume,
        },

        # Evaluation
        "eval": {
            "n_episodes": 5,
            "batch_size": 1,
        },

        # Task language conditioning
        "task_description": TASK_DESCRIPTION,

        # Normalisation (will be computed from data if not provided)
        "normalisation": "mean_std",
    }
    return config


# ---------------------------------------------------------------------------
# Run training via LeRobot CLI
# ---------------------------------------------------------------------------

def run_lerobot_train(config: dict, dry_run: bool) -> None:
    """
    Write the config to a temporary YAML and invoke lerobot-train (or
    the Python training module directly).
    """
    import subprocess
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="tinyvla_train_"
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        cfg_path = f.name

    logger.info(f"Config written to: {cfg_path}")

    # Try lerobot-train CLI first, then fall back to module
    cmds_to_try = [
        ["lerobot-train", f"--config={cfg_path}"],
        ["python", "-m", "lerobot.scripts.train", f"--config={cfg_path}"],
        ["python", "-m", "lerobot.train", f"--config={cfg_path}"],
    ]

    if dry_run:
        print("\n[dry-run] Would run one of:\n")
        for cmd in cmds_to_try:
            print("  ", " ".join(cmd))
        print(f"\nConfig preview ({cfg_path}):")
        with open(cfg_path) as f:
            print(f.read())
        return

    for cmd in cmds_to_try:
        try:
            result = subprocess.run(cmd, check=True)
            logger.info("Training completed successfully.")
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed: {' '.join(cmd)}\n  Exit code: {e.returncode} — trying native loop")
            break

    # Neither worked — fall back to native Python training
    logger.warning("lerobot-train CLI not found. Attempting native Python training...")
    run_native_train(config, dry_run=False)


# ---------------------------------------------------------------------------
# Native training loop (runs without lerobot-train CLI)
# ---------------------------------------------------------------------------

def run_native_train(config: dict, dry_run: bool = False) -> None:
    """
    Minimal training loop using LeRobot Python APIs directly.
    Mirrors what lerobot.scripts.train does but with our config dict.
    """
    try:
        import torch
        # Try LeRobot v2 paths first, fall back to v1
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.act.configuration_act import ACTConfig
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.common.policies.act.configuration_act import ACTConfig
    except ImportError as e:
        print(f"\nERROR: Cannot import LeRobot: {e}")
        print("Install LeRobot: pip install git+https://github.com/huggingface/lerobot")
        sys.exit(1)

    device = torch.device(config["device"])
    logger.info(f"Training on device: {device}")

    # ---- Dataset -----------------------------------------------------------
    datasets = []
    for repo_id, root in zip(
        config["dataset"]["repo_ids"],
        config["dataset"]["roots"]
    ):
        full_path = str(Path(root) / repo_id)
        try:
            # Try local-only load with full path as root (LeRobot v2)
            try:
                ds = LeRobotDataset(repo_id=repo_id, root=full_path, local_files_only=True)
            except TypeError:
                # local_files_only not supported in this version — use root only
                ds = LeRobotDataset(repo_id=repo_id, root=full_path)
            datasets.append(ds)
            logger.info(f"  Loaded dataset: {repo_id} ({len(ds)} frames)")
        except Exception as ex:
            logger.warning(f"  Could not load {repo_id} from {full_path}: {ex}")

    if not datasets:
        logger.error("No datasets could be loaded. Check dataset paths.")
        sys.exit(1)

    # Concatenate datasets if multiple
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets)
        logger.info(f"Combined dataset: {len(dataset)} total frames")

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ---- Policy ------------------------------------------------------------
    pc = config["policy"]
    policy_cfg = ACTConfig(
        chunk_size              = pc["chunk_size"],
        n_action_steps          = pc["n_action_steps"],
        n_obs_steps             = pc["n_obs_steps"],
        temporal_ensemble_coeff = pc["temporal_ensemble_coeff"],
        vision_backbone         = pc["vision_backbone"],
        pretrained_backbone_weights = pc["pretrained_backbone_weights"],
        dim_model               = pc["dim_model"],
        n_heads                 = pc["n_heads"],
        dim_feedforward         = pc["dim_feedforward"],
        n_encoder_layers        = pc["n_encoder_layers"],
        n_decoder_layers        = pc["n_decoder_layers"],
        use_vae                 = pc["use_vae"],
        latent_dim              = pc["latent_dim"],
        n_vae_encoder_layers    = pc["n_vae_encoder_layers"],
        kl_weight               = pc["kl_weight"],
    )

    # Resume or create fresh
    resume_from = config["training"].get("resume_from")
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        policy = ACTPolicy.from_pretrained(resume_from)
    else:
        policy = ACTPolicy(policy_cfg, dataset_stats=getattr(dataset, "stats", None))

    policy = policy.to(device)

    # ---- Optimiser ---------------------------------------------------------
    backbone_params = []
    other_params    = []
    for name, param in policy.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": other_params,    "lr": config["training"]["lr"]},
        {"params": backbone_params, "lr": config["training"]["lr_backbone"]},
    ], weight_decay=config["training"]["weight_decay"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[dry-run] Policy and data loaders created successfully. Exiting.")
        return

    # ---- Training loop -----------------------------------------------------
    epochs  = config["training"]["num_epochs"]
    log_every  = config["training"]["log_every"]
    save_every = config["training"]["save_checkpoint_every"]

    global_step = 0
    best_loss   = float("inf")

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"  Batch size : {config['training']['batch_size']}")
    logger.info(f"  LR         : {config['training']['lr']}")
    logger.info(f"  Chunk size : {pc['chunk_size']}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Task       : {config['task_description']}")

    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            # Move to device
            batch = {k: v.to(device) if hasattr(v, "to") else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            loss, loss_dict = policy.forward(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config["training"]["grad_clip_norm"]
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if global_step % log_every == 0:
                logger.info(
                    f"  epoch={epoch+1}/{epochs}  step={global_step}"
                    f"  loss={loss.item():.4f}"
                    f"  kl={loss_dict.get('kl_loss', 0):.4f}"
                    f"  recon={loss_dict.get('reconstruction_loss', 0):.4f}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or avg_loss < best_loss:
            ckpt_dir = output_dir / f"epoch_{epoch+1:04d}"
            policy.save_pretrained(str(ckpt_dir))
            logger.info(f"  Checkpoint saved → {ckpt_dir}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_dir = output_dir / "best"
                policy.save_pretrained(str(best_dir))
                logger.info(f"  Best checkpoint updated → {best_dir}  (loss={best_loss:.4f})")

    # Final save
    final_dir = output_dir / "final"
    policy.save_pretrained(str(final_dir))
    logger.info(f"\nTraining complete. Final model → {final_dir}")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    logger.info(f"\nTo use this model, set the checkpoint path in robot_api.py or run:")
    logger.info(f"  python run.py --model-path {final_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_tinyvla",
        description="Fine-tune TinyVLA/ACT on SO101 pick-and-place data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        metavar="PATH",
        help="Explicit dataset paths. If omitted, auto-discovers from ./data/ and ./skillpatch_data/",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Checkpoint output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"ACT action chunk horizon steps (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "rocm"],
        help="Compute device (default: auto — prefers ROCm/CUDA, falls back to CPU)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT_DIR",
        help="Resume training from a checkpoint directory",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show config and dataset list without training",
    )
    parser.add_argument(
        "--native", action="store_true",
        help="Use native Python training loop instead of lerobot-train CLI",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Explicit ROCm check — HIP version present means AMD GPU is available
        if getattr(torch.version, "hip", None) is not None:
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n══════════════════════════════════════════════")
    print("  TinyVLA Fine-Tuning — Pick-and-Place (SO101)")
    print("══════════════════════════════════════════════\n")

    # ---- Resolve datasets ---------------------------------------------------
    if args.datasets:
        datasets = args.datasets
        logger.info(f"Using explicitly specified datasets ({len(datasets)}):")
        for ds in datasets:
            if not dataset_exists(ds):
                logger.error(f"Dataset not found: {ds}")
                sys.exit(1)
            n = count_episodes(ds)
            logger.info(f"  ✓ {ds} ({n} episodes)")
    else:
        logger.info("Auto-discovering datasets...")
        datasets = discover_datasets()

    if not datasets:
        print("\nERROR: No datasets found.")
        print("Record demonstrations first:")
        print("  python record_episode.py --action full_pick_and_place")
        print("  python record_episode.py --action pick_object")
        print("  python record_episode.py --action place_in_shelf")
        sys.exit(1)

    total_episodes = sum(count_episodes(ds) for ds in datasets)
    logger.info(f"\nTotal: {len(datasets)} dataset(s), {total_episodes} episode(s)")

    if total_episodes < 10:
        logger.warning(
            f"Only {total_episodes} episodes found. "
            "For reliable pick-and-place training, aim for ≥ 50 full-motion episodes. "
            "The model will train but may not generalise well."
        )

    # ---- Build config -------------------------------------------------------
    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    config = build_lerobot_config(
        datasets   = datasets,
        output_dir = args.output,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        chunk_size = args.chunk_size,
        device     = device,
        seed       = args.seed,
        resume     = args.resume,
    )

    logger.info(f"\nTraining config summary:")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Batch size : {args.batch_size}")
    logger.info(f"  LR         : {args.lr}")
    logger.info(f"  Chunk size : {args.chunk_size} steps")
    logger.info(f"  Output     : {args.output}")
    logger.info(f"  Task       : {TASK_DESCRIPTION}")

    # ---- Run ----------------------------------------------------------------
    if args.native:
        run_native_train(config, dry_run=args.dry_run)
    else:
        run_lerobot_train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
