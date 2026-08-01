#!/usr/bin/env python3
"""Plot value curve aligned with robot actions for bowls_tray v2 (or any tagged dataset).

Uses offline annotations from value-infer (already on the merged train set).
For a fresh pure-policy eval recording, run value-infer on that eval root first,
then point --root at it.

Examples:
  python plot_value_action_curve.py --episodes 50,200,220
  python plot_value_action_curve.py --episodes 150-155 --out /tmp/val_curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_episodes(arg: str, total: int) -> list[int]:
    out: set[int] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    eps = sorted(out)
    for e in eps:
        if e < 0 or e >= total:
            raise SystemExit(f"episode {e} out of range [0,{total})")
    return eps


def _stack_vec(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float32)


def plot_episode(
    df: pd.DataFrame,
    ep: int,
    success: str,
    value_key: str,
    adv_key: str | None,
    acp_key: str | None,
    out_path: Path,
    fps: float,
) -> None:
    t = df["frame_index"].to_numpy().astype(np.float32) / fps
    value = df[value_key].to_numpy().astype(np.float32)
    action = _stack_vec(df["action"])
    state = _stack_vec(df["observation.state"])
    act_norm = np.linalg.norm(action, axis=1)
    residual = np.linalg.norm(action - state, axis=1)
    inter = (
        df["complementary_info.is_intervention"].to_numpy().astype(np.float32)
        if "complementary_info.is_intervention" in df.columns
        else None
    )
    if "complementary_info.policy_action" in df.columns:
        pol = _stack_vec(df["complementary_info.policy_action"])
        pol_gap = np.linalg.norm(action - pol, axis=1)
    else:
        pol_gap = None

    n_panels = 3 + (1 if pol_gap is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.2 * n_panels), sharex=True)
    fig.suptitle(f"ep{ep}  success={success}  frames={len(df)}  ({t[-1]:.1f}s)", fontsize=12)

    ax = axes[0]
    ax.plot(t, value, color="#1f77b4", lw=1.5, label="value")
    if adv_key and adv_key in df.columns:
        adv = df[adv_key].to_numpy().astype(np.float32)
        ax2 = ax.twinx()
        ax2.plot(t, adv, color="#ff7f0e", alpha=0.7, lw=1.0, label="advantage")
        ax2.set_ylabel("advantage", color="#ff7f0e")
    if acp_key and acp_key in df.columns:
        acp = df[acp_key].to_numpy().astype(np.float32)
        ymin, ymax = float(np.nanmin(value)), float(np.nanmax(value))
        ax.fill_between(t, ymin, ymax, where=acp > 0.5, color="#2ca02c", alpha=0.12, label="ACP+")
    if inter is not None:
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t, ymin, ymax, where=inter > 0.5, color="#d62728", alpha=0.15, label="human i")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, act_norm, color="#9467bd", lw=1.2, label="||action||")
    ax.plot(t, residual, color="#8c564b", lw=1.2, label="||action-state|| (hold≈0)")
    ax.set_ylabel("action mag")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    # left 6 / right 6 joints
    for j in range(min(6, action.shape[1])):
        ax.plot(t, action[:, j], lw=0.8, alpha=0.85, label=f"L{j}" if j < 3 else None)
    for j in range(6, min(12, action.shape[1])):
        ax.plot(t, action[:, j], lw=0.8, alpha=0.85, ls="--", label=f"R{j-6}" if j < 9 else None)
    ax.set_ylabel("joint action")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    if pol_gap is not None:
        ax = axes[3]
        ax.plot(t, pol_gap, color="#e377c2", lw=1.2, label="||exec - policy_action||")
        if inter is not None:
            ax.fill_between(t, 0, pol_gap.max(initial=1.0), where=inter > 0.5, color="#d62728", alpha=0.15)
        ax.set_ylabel("policy gap")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v2",
    )
    ap.add_argument("--repo-id", default="my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v2")
    ap.add_argument("--tag", default="bowls_tray_3cam_v2", help="suffix of complementary_info.* fields")
    ap.add_argument("--episodes", required=True, help="e.g. 50,200 or 150-155")
    ap.add_argument(
        "--out",
        default="/home/rxn/lerobot/Evo-RL/outputs/eval/value_action_curves_v2",
    )
    args = ap.parse_args()

    root = Path(args.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    total = int(info["total_episodes"])
    episodes = _parse_episodes(args.episodes, total)

    value_key = f"complementary_info.value_{args.tag}"
    adv_key = f"complementary_info.advantage_{args.tag}"
    acp_key = f"complementary_info.acp_indicator_{args.tag}"
    if value_key not in info["features"]:
        raise SystemExit(f"missing {value_key} in dataset features; available: "
                         f"{[k for k in info['features'] if 'value' in k]}")

    edf = pd.concat(
        [pd.read_parquet(f) for f in (root / "meta" / "episodes").rglob("*.parquet")],
        ignore_index=True,
    )
    # single data file for this dataset
    data_files = sorted((root / "data").rglob("*.parquet"))
    if len(data_files) != 1:
        # fallback: load needed ranges per file via dataset_from_index mapping
        raise SystemExit(f"expected 1 data parquet, found {len(data_files)}")
    print(f"loading {data_files[0]} ...")
    full = pd.read_parquet(data_files[0])

    out_dir = Path(args.out)
    for ep in episodes:
        row = edf[edf.episode_index == ep].iloc[0]
        a, b = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        success = str(row["episode_success"]) if "episode_success" in row.index else "?"
        ep_df = full.iloc[a:b].reset_index(drop=True)
        plot_episode(
            ep_df,
            ep=ep,
            success=success,
            value_key=value_key,
            adv_key=adv_key,
            acp_key=acp_key,
            out_path=out_dir / f"ep{ep:03d}_{success}_value_action.png",
            fps=fps,
        )
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
