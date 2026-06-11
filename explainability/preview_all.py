"""
Generate preview images for ALL preprocessed LAformer samples.

Each preview shows: ego history, ground truth, all 5 predicted modes,
neighbor agents (color-coded by type), and lane polylines.

Usage:
    cd <project_root>
    conda activate laformer-explain

    # Validation set (61 samples):
    python -m explainability.preview_all

    # Training set (742 samples):
    python -m explainability.preview_all \
        --cache_path data/mini_train_cache/ex_list \
        --output_dir results/previews/train
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "laformer"))
sys.path.insert(0, str(PROJ_ROOT))

from explainability.explain_utils import (
    load_model,
    load_samples,
    predict,
    classify_neighbors,
    compute_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODE_COLORS = ["#2D93AD", "#27AE60", "#F39C12", "#9B59B6", "#E74C3C"]
VEH_COLOR = "#6B8E23"
PED_COLOR = "#D35400"
OTHER_COLOR = "#7F8C8D"


def generate_preview(mapping, neighbor_info, preds, scores, gt_rel, metrics,
                     scene_name, idx, out_path):
    """Generate a single preview image for one sample."""
    past = np.array(mapping["past_traj"])[:, :2]
    lanes = mapping.get("polygons", [])
    n_agents = len(mapping["trajs"])
    n_lanes = len(mapping["polyline_spans"]) - mapping["map_start_polyline_idx"]
    n_veh = sum(1 for n in neighbor_info if n["type"] == "Vehicle")
    n_ped = sum(1 for n in neighbor_info if n["type"] == "Pedestrian")
    n_other = sum(1 for n in neighbor_info if n["type"] not in ("Vehicle", "Pedestrian"))
    n_mislabeled = sum(1 for n in neighbor_info if n["mislabeled"])

    min_fde = min(
        float(np.linalg.norm(preds[k][-1] - gt_rel[-1]))
        for k in range(len(scores))
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for lane in lanes:
        l = np.array(lane)
        ax.plot(l[:, 0], l[:, 1], color="#CCCCCC", lw=0.8, alpha=0.6, zorder=1)

    for info in neighbor_info:
        t = info["traj"]
        if info["type"] == "Vehicle":
            color, mk, alpha = VEH_COLOR, "s", 0.5
        elif info["type"] == "Pedestrian":
            color, mk, alpha = PED_COLOR, "o", 0.7
        else:
            color, mk, alpha = OTHER_COLOR, "^", 0.5
        ax.plot(t[:, 0], t[:, 1], color=color, lw=1.5, alpha=alpha, zorder=3)
        ax.plot(t[-1, 0], t[-1, 1], mk, color=color, ms=5, alpha=0.6, zorder=4,
                mec="white", mew=0.3)

    ax.plot(past[:, 0], past[:, 1], "--", color="#333333", lw=2.5, zorder=7)
    ax.plot(past[-1, 0], past[-1, 1], "o", color="#333333", ms=10, zorder=8)

    ax.plot(gt_rel[:, 0], gt_rel[:, 1], color="#E74C8B", lw=3.5, alpha=0.9, zorder=9,
            path_effects=[pe.withStroke(linewidth=5.5, foreground="white", alpha=0.4)])
    ax.plot(gt_rel[-1, 0], gt_rel[-1, 1], "*", color="#E74C8B", ms=14, zorder=11,
            mec="white", mew=0.8)

    sorted_m = np.argsort(scores)[::-1]
    for mi in sorted_m:
        is_best = (mi == sorted_m[0])
        ax.plot(preds[mi, :, 0], preds[mi, :, 1],
                color=MODE_COLORS[mi % len(MODE_COLORS)],
                lw=2.5 if is_best else 1.5,
                alpha=0.9 if is_best else 0.45,
                zorder=10 if is_best else 8)
        ax.plot(preds[mi, -1, 0], preds[mi, -1, 1], "o",
                color=MODE_COLORS[mi % len(MODE_COLORS)],
                ms=7 if is_best else 4, alpha=0.8, zorder=10)

    legend_els = [
        Line2D([0], [0], color="#333333", ls="--", lw=2, label="Ego History"),
        Line2D([0], [0], color="#E74C8B", lw=3, label="Ground Truth"),
        Line2D([0], [0], color=MODE_COLORS[0], lw=2.5, label="Best Mode"),
        Line2D([0], [0], color=MODE_COLORS[1], lw=1.5, alpha=0.5, label="Other Modes"),
        Line2D([0], [0], color=VEH_COLOR, lw=1.5, marker="s", ms=5, label=f"Vehicles ({n_veh})"),
        Line2D([0], [0], color=PED_COLOR, lw=1.5, marker="o", ms=5, label=f"Pedestrians ({n_ped})"),
    ]
    if n_other > 0:
        legend_els.append(
            Line2D([0], [0], color=OTHER_COLOR, lw=1.5, marker="^", ms=5, label=f"Other ({n_other})")
        )
    ax.legend(handles=legend_els, loc="upper right", fontsize=9, framealpha=0.9)

    mislabel_note = f" ({n_mislabeled} mislabeled)" if n_mislabeled else ""
    ax.set_title(
        f"LAformer Preview: {scene_name} idx={idx}\n"
        f"{n_agents} agents ({n_veh}V + {n_ped}P{mislabel_note}) | {n_lanes} lanes\n"
        f"ADE={metrics['ade']:.2f}m  FDE={metrics['fde']:.2f}m  minFDE={min_fde:.2f}m",
        fontsize=13, fontweight="bold", pad=12,
    )

    all_pts = np.concatenate([past, gt_rel, preds[sorted_m[0]]])
    xmn, xmx = all_pts[:, 0].min() - 8, all_pts[:, 0].max() + 8
    ymn, ymx = all_pts[:, 1].min() - 8, all_pts[:, 1].max() + 8
    rng = max(xmx - xmn, ymx - ymn)
    mx, my = (xmn + xmx) / 2, (ymn + ymx) / 2
    ax.set_xlim(mx - rng / 2, mx + rng / 2)
    ax.set_ylim(my - rng / 2, my + rng / 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LAformer preview images for all samples")
    parser.add_argument("--cache_path", type=str,
                        default=str(PROJ_ROOT / "data" / "mini_val_cache" / "eval.ex_list"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--nusc_datadir", type=str,
                        default=str(PROJ_ROOT / "nuscenes_mini"))
    parser.add_argument("--nusc_version", type=str, default="v1.0-mini")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJ_ROOT / "results" / "previews" / "val"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading LAformer model...")
    model, _ = load_model(args.checkpoint, args.device)

    logger.info("Loading samples from %s", args.cache_path)
    samples = load_samples(args.cache_path)

    logger.info("Loading nuScenes for GT annotations...")
    from nuscenes import NuScenes
    nusc = NuScenes(args.nusc_version, dataroot=args.nusc_datadir, verbose=False)

    logger.info("Generating previews for %d samples...", len(samples))
    summary_rows = []

    for idx, mapping in enumerate(samples):
        sample_rec = nusc.get("sample", mapping["sample_token"])
        scene_rec = nusc.get("scene", sample_rec["scene_token"])
        scene_name = scene_rec["name"]

        gt_rel = np.array(mapping["labels"])
        preds, scores = predict(model, mapping, device)
        best = preds[np.argmax(scores)]
        metrics = compute_metrics(best, gt_rel)
        min_fde = min(
            float(np.linalg.norm(preds[k][-1] - gt_rel[-1]))
            for k in range(len(scores))
        )

        neighbor_info = classify_neighbors(mapping, nusc)
        n_veh = sum(1 for n in neighbor_info if n["type"] == "Vehicle")
        n_ped = sum(1 for n in neighbor_info if n["type"] == "Pedestrian")
        n_agents = len(mapping["trajs"])
        n_mislabeled = sum(1 for n in neighbor_info if n["mislabeled"])

        fname = f"{idx:03d}_{scene_name}_agents{n_agents}_fde{metrics['fde']:.1f}.png"
        out_path = os.path.join(args.output_dir, fname)
        generate_preview(mapping, neighbor_info, preds, scores, gt_rel, metrics,
                         scene_name, idx, out_path)

        summary_rows.append({
            "idx": idx,
            "scene": scene_name,
            "n_agents": n_agents,
            "n_vehicles": n_veh,
            "n_pedestrians": n_ped,
            "n_mislabeled": n_mislabeled,
            "ade": round(metrics["ade"], 2),
            "fde": round(metrics["fde"], 2),
            "min_fde": round(min_fde, 2),
            "file": fname,
        })

        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            logger.info("  %d / %d done", idx + 1, len(samples))

    import json
    summary_path = os.path.join(args.output_dir, "preview_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    fdes = [r["fde"] for r in summary_rows]
    logger.info("FDE stats: min=%.2f, median=%.2f, mean=%.2f, max=%.2f",
                min(fdes), np.median(fdes), np.mean(fdes), max(fdes))
    logger.info("All %d previews saved to %s", len(samples), args.output_dir)


if __name__ == "__main__":
    main()
