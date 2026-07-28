#!/usr/bin/env python3
"""Merge bowls_tray train (strip old value tags) + valid HITL runs -> v2 dataset."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TRAIN_ROOT = Path("/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue")
TRAIN_REPO = "my_bimanual/evo_rl_bowls_stack_lipstick_tissue"
HITL_ROOT = Path("/home/rxn/lerobot/Evo-RL/outputs/eval/hitl_bowls_tray_3cam")
STRIP_ROOT = Path("/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_novalue")
STRIP_REPO = "my_bimanual/evo_rl_bowls_stack_lipstick_tissue_novalue"
OUT_ROOT = Path("/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v2")
OUT_REPO = "my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v2"

VALUE_FEATURES = [
    "complementary_info.value_bowls_tray_3cam",
    "complementary_info.advantage_bowls_tray_3cam",
    "complementary_info.acp_indicator_bowls_tray_3cam",
]


def valid_hitl_dirs() -> list[Path]:
    dirs = []
    for d in sorted(HITL_ROOT.iterdir()):
        info = d / "meta" / "info.json"
        if not info.exists():
            continue
        eps = json.loads(info.read_text()).get("total_episodes", 0)
        if eps and eps > 0:
            dirs.append(d)
    return dirs


def main() -> int:
    hitl_dirs = valid_hitl_dirs()
    if not hitl_dirs:
        print("ERROR: no valid HITL datasets", file=sys.stderr)
        return 1

    train_info = json.loads((TRAIN_ROOT / "meta" / "info.json").read_text())
    print(f"train: {train_info['total_episodes']} eps / {train_info['total_frames']} frames")
    hitl_eps = 0
    for d in hitl_dirs:
        i = json.loads((d / "meta" / "info.json").read_text())
        print(f"  HITL {d.name}: {i['total_episodes']} eps")
        hitl_eps += i["total_episodes"]
    print(f"HITL total: {hitl_eps} eps -> expect ~{train_info['total_episodes'] + hitl_eps} after merge")

    if OUT_ROOT.exists():
        print(f"Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    if STRIP_ROOT.exists():
        print(f"Removing existing {STRIP_ROOT}")
        shutil.rmtree(STRIP_ROOT)

    print("=== strip old value/ACP features from train ===")
    train_ds = LeRobotDataset(TRAIN_REPO, root=str(TRAIN_ROOT))
    present = [f for f in VALUE_FEATURES if f in train_ds.meta.features]
    if present:
        strip_ds = remove_feature(
            train_ds,
            feature_names=present,
            output_dir=str(STRIP_ROOT),
            repo_id=STRIP_REPO,
        )
        used_strip_tmp = True
    else:
        print("No value features to strip; using train root directly")
        strip_ds = train_ds
        used_strip_tmp = False

    print(
        f"stripped: {strip_ds.meta.total_episodes} eps, "
        f"features complementary={[k for k in strip_ds.meta.features if 'complementary' in k]}"
    )

    print("=== merge train_stripped + HITL ===")
    datasets = [strip_ds]
    for i, d in enumerate(hitl_dirs):
        # unique repo_id per source so aggregate doesn't confuse roots
        repo = f"eval/hitl_bowls_tray_merge_src_{i}"
        datasets.append(LeRobotDataset(repo, root=str(d)))

    merged = merge_datasets(datasets, output_repo_id=OUT_REPO, output_dir=str(OUT_ROOT))
    print(f"MERGED -> {OUT_ROOT}")
    print(f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}")

    # cleanup intermediate to free disk
    if used_strip_tmp and STRIP_ROOT.exists():
        print(f"cleanup {STRIP_ROOT}")
        shutil.rmtree(STRIP_ROOT)

    out_info = json.loads((OUT_ROOT / "meta" / "info.json").read_text())
    print("DONE", json.dumps({"episodes": out_info["total_episodes"], "frames": out_info["total_frames"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
