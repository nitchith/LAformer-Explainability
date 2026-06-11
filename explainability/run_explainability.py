"""
LAformer Explainability Analysis: Neighbor Ablation + Temporal Occlusion.

Reusable script to run explainability on any LAformer sample.

Usage:
    cd <project_root>
    conda activate laformer-explain

    # By sample index (val set):
    python -m explainability.run_explainability --sample_idx 23

    # By scene name (picks the sample with most agents):
    python -m explainability.run_explainability --scene scene-0103

    # On a training set sample:
    python -m explainability.run_explainability \
        --sample_idx 545 \
        --cache_path data/mini_train_cache/ex_list

    # Skip temporal occlusion (faster):
    python -m explainability.run_explainability \
        --sample_idx 23 --skip_temporal
"""

import argparse
import copy
import json
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
    remove_neighbors,
    remove_lanes,
    remove_lanes_and_neighbors,
    compute_metrics,
    compute_shift,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODE_COLORS = ["#2D93AD", "#27AE60", "#F39C12", "#9B59B6", "#E74C3C"]
VEH_COLOR = "#6B8E23"
PED_COLOR = "#D35400"


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------

def run_neighbor_ablation(model, mapping, device, neighbor_info, baseline_best):
    """Run individual leave-one-out ablation for every neighbor."""
    results = []
    for ni in range(len(mapping["trajs"])):
        m_abl = remove_neighbors(mapping, [ni])
        pr, sc = predict(model, m_abl, device)
        pb = pr[np.argmax(sc)]
        shift = compute_shift(baseline_best, pb)
        info = neighbor_info[ni]
        results.append({
            "idx": ni, "type": info["type"], "cat": info["cat"],
            "dist": info["dist"], "pred": pb, "all_modes": pr, "scores": sc,
            **shift,
        })
    results.sort(key=lambda x: x["ade_shift"], reverse=True)
    return results


def run_group_ablation(model, mapping, device, neighbor_info, baseline_best, gt_rel):
    """Remove pedestrians / vehicles / all / lanes and measure shift."""
    groups = {}

    ped_idx = [n["idx"] for n in neighbor_info if n["type"] == "Pedestrian"]
    veh_idx = [n["idx"] for n in neighbor_info if n["type"] == "Vehicle"]
    all_idx = list(range(len(mapping["trajs"])))

    for label, idx_list in [
        ("no_peds", ped_idx),
        ("no_vehs", veh_idx),
        ("no_agents", all_idx),
    ]:
        if not idx_list:
            continue
        m_abl = remove_neighbors(mapping, idx_list)
        pr, sc = predict(model, m_abl, device)
        best = pr[np.argmax(sc)]
        min_fde = min(
            float(np.linalg.norm(pr[k][-1] - gt_rel[-1]))
            for k in range(len(sc))
        )
        groups[label] = {
            "pred": best,
            "all_modes": pr,
            "scores": sc,
            "n_removed": len(idx_list),
            "min_fde": min_fde,
            **compute_metrics(best, gt_rel),
            **compute_shift(baseline_best, best),
        }

    n_lanes = len(mapping["polyline_spans"]) - mapping["map_start_polyline_idx"]
    m_no_lanes = remove_lanes(mapping)
    pr, sc = predict(model, m_no_lanes, device)
    best = pr[np.argmax(sc)]
    groups["no_lanes"] = {
        "pred": best,
        "all_modes": pr,
        "scores": sc,
        "n_removed": n_lanes,
        "min_fde": min(float(np.linalg.norm(pr[k][-1] - gt_rel[-1])) for k in range(len(sc))),
        **compute_metrics(best, gt_rel),
        **compute_shift(baseline_best, best),
    }

    m_ego_only = remove_lanes_and_neighbors(mapping)
    pr, sc = predict(model, m_ego_only, device)
    best = pr[np.argmax(sc)]
    groups["ego_only"] = {
        "pred": best,
        "all_modes": pr,
        "scores": sc,
        "n_removed": n_lanes + len(all_idx),
        "min_fde": min(float(np.linalg.norm(pr[k][-1] - gt_rel[-1])) for k in range(len(sc))),
        **compute_metrics(best, gt_rel),
        **compute_shift(baseline_best, best),
    }

    return groups


def run_temporal_occlusion(model, mapping, device, baseline_best):
    """Mask each ego history timestep and measure prediction shift."""
    past = np.array(mapping["past_traj"])[:, :2]
    num_hist = len(past)
    results = []
    for t in range(num_hist):
        m_copy = copy.deepcopy(mapping)
        ego_span = m_copy["polyline_spans"][0]
        if t < (ego_span.stop - ego_span.start):
            m_copy["matrix"][ego_span.start + t, :5] = 0.0
        pr, sc = predict(model, m_copy, device)
        best = pr[np.argmax(sc)]
        shift = compute_shift(baseline_best, best)
        results.append({
            "t": t,
            "recency": num_hist - 1 - t,
            "pred": best,
            **shift,
        })
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_agents(ax, neighbor_info, removed_type=None):
    for info in neighbor_info:
        t = info["traj"]
        is_rm = removed_type == "ALL" or (
            removed_type and info["type"] == removed_type
        )
        color = VEH_COLOR if info["type"] == "Vehicle" else (
            PED_COLOR if info["type"] == "Pedestrian" else "gray"
        )
        mk = "s" if info["type"] == "Vehicle" else "o"

        if is_rm:
            ax.plot(t[:, 0], t[:, 1], color=color, lw=0.8, alpha=0.12, ls=":", zorder=2)
            ax.plot(t[-1, 0], t[-1, 1], "x", color="red", ms=7, alpha=0.4, zorder=3, markeredgewidth=2)
        else:
            alpha = 0.7 if info["type"] == "Pedestrian" else 0.5
            ax.plot(t[:, 0], t[:, 1], color=color, lw=1.5, alpha=alpha, zorder=2 + (2 if info["type"] == "Pedestrian" else 0))
            ax.plot(t[-1, 0], t[-1, 1], mk, color=color, ms=5, alpha=0.6, zorder=3, mec="white", mew=0.3)


def _draw_ego_and_gt(ax, past, gt_rel):
    ax.plot(past[:, 0], past[:, 1], "--", color="#333333", lw=2.5, zorder=7)
    ax.plot(past[-1, 0], past[-1, 1], "o", color="#333333", ms=10, zorder=8)
    ax.plot(gt_rel[:, 0], gt_rel[:, 1], color="#E74C8B", lw=3.5, alpha=0.9, zorder=9,
            path_effects=[pe.withStroke(linewidth=5.5, foreground="white", alpha=0.4)])
    ax.plot(gt_rel[-1, 0], gt_rel[-1, 1], "*", color="#E74C8B", ms=14, zorder=11,
            mec="white", mew=0.8)


def _draw_lanes(ax, lanes):
    for lane in lanes:
        l = np.array(lane)
        ax.plot(l[:, 0], l[:, 1], color="#CCCCCC", lw=0.8, alpha=0.6, zorder=1)


def _draw_modes(ax, preds, scores):
    sorted_m = np.argsort(scores)[::-1]
    for mi in sorted_m:
        ib = mi == sorted_m[0]
        ax.plot(preds[mi, :, 0], preds[mi, :, 1],
                color=MODE_COLORS[mi % len(MODE_COLORS)],
                lw=2.5 if ib else 1.5, alpha=0.9 if ib else 0.45,
                zorder=10 if ib else 8)
        ax.plot(preds[mi, -1, 0], preds[mi, -1, 1], "o",
                color=MODE_COLORS[mi % len(MODE_COLORS)],
                ms=7 if ib else 4, alpha=0.8, zorder=10)


def _auto_lim(ax, *point_arrays):
    all_pts = np.concatenate(point_arrays)
    xmn, xmx = all_pts[:, 0].min() - 8, all_pts[:, 0].max() + 8
    ymn, ymx = all_pts[:, 1].min() - 8, all_pts[:, 1].max() + 8
    rng = max(xmx - xmn, ymx - ymn)
    mx, my = (xmn + xmx) / 2, (ymn + ymx) / 2
    ax.set_xlim(mx - rng / 2, mx + rng / 2)
    ax.set_ylim(my - rng / 2, my + rng / 2)


def plot_six_panel(
    mapping, neighbor_info, baseline, groups, out_dir, scene_label,
):
    """Baseline vs no-peds vs no-vehs vs no-agents vs no-lanes vs ego-only."""
    past = np.array(mapping["past_traj"])[:, :2]
    gt_rel = np.array(mapping["labels"])
    lanes = mapping.get("polygons", [])
    n_veh = sum(1 for n in neighbor_info if n["type"] == "Vehicle")
    n_ped = sum(1 for n in neighbor_info if n["type"] == "Pedestrian")
    n_lanes = len(mapping["polyline_spans"]) - mapping["map_start_polyline_idx"]

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    axes = axes.flatten()

    base_m = compute_metrics(baseline["best"], gt_rel)
    base_minfde = min(
        float(np.linalg.norm(baseline["all_modes"][k][-1] - gt_rel[-1]))
        for k in range(len(baseline["scores"]))
    )

    panels = [
        ("(a) Baseline", baseline["all_modes"], baseline["scores"], None, False,
         f'ADE={base_m["ade"]:.2f}m  FDE={base_m["fde"]:.2f}m  minFDE={base_minfde:.2f}m'),
    ]
    panel_defs = [
        ("no_peds", "Pedestrian", False, "(b) Pedestrians Removed"),
        ("no_vehs", "Vehicle", False, "(c) Vehicles Removed"),
        ("no_agents", "ALL", False, "(d) All Agents Removed"),
        ("no_lanes", None, True, "(e) All Lanes Removed"),
        ("ego_only", "ALL", True, "(f) Ego Only (no agents, no lanes)"),
    ]
    for key, rm_type, hide_lanes, label in panel_defs:
        if key in groups:
            g = groups[key]
            panels.append((
                label, g["all_modes"], g["scores"], rm_type, hide_lanes,
                f'ADE={g["ade"]:.2f}m  FDE={g["fde"]:.2f}m\nShift: {g["ade_shift"]:.3f}m',
            ))

    for ci, (title, preds, scrs, rm_type, hide_lanes, metrics) in enumerate(panels):
        if ci >= len(axes):
            break
        ax = axes[ci]

        if not hide_lanes:
            _draw_lanes(ax, lanes)
        else:
            for lane in lanes:
                l = np.array(lane)
                ax.plot(l[:, 0], l[:, 1], color="#CCCCCC", lw=0.5, alpha=0.15, ls=":", zorder=1)

        _draw_agents(ax, neighbor_info, removed_type=rm_type)
        _draw_ego_and_gt(ax, past, gt_rel)
        _draw_modes(ax, preds, scrs)
        if ci > 0:
            ax.plot(baseline["best"][:, 0], baseline["best"][:, 1],
                    ":", color="#2D93AD", lw=1.5, alpha=0.5, zorder=7)

        legend_els = [
            Line2D([0], [0], color="#333333", ls="--", lw=2, label="Ego History"),
            Line2D([0], [0], color="#E74C8B", lw=3, label="Ground Truth"),
        ]
        if rm_type == "Pedestrian":
            legend_els.append(Line2D([0], [0], color=VEH_COLOR, lw=1.5, marker="s", ms=5, label=f"Vehicles ({n_veh})"))
            legend_els.append(Line2D([0], [0], color="red", lw=0, marker="x", ms=8, label=f"Peds REMOVED ({n_ped})"))
        elif rm_type == "Vehicle":
            legend_els.append(Line2D([0], [0], color="red", lw=0, marker="x", ms=8, label=f"Vehs REMOVED ({n_veh})"))
            legend_els.append(Line2D([0], [0], color=PED_COLOR, lw=2, marker="o", ms=5, label=f"Pedestrians ({n_ped})"))
        elif rm_type == "ALL":
            legend_els.append(Line2D([0], [0], color="red", lw=0, marker="x", ms=8, label=f"Agents REMOVED ({len(mapping['trajs'])})"))
        else:
            legend_els.append(Line2D([0], [0], color=VEH_COLOR, lw=1.5, marker="s", ms=5, label=f"Vehicles ({n_veh})"))
            legend_els.append(Line2D([0], [0], color=PED_COLOR, lw=2, marker="o", ms=5, label=f"Pedestrians ({n_ped})"))
        if hide_lanes:
            legend_els.append(Line2D([0], [0], color="#CCCCCC", ls=":", lw=1, label=f"Lanes REMOVED ({n_lanes})"))
        if ci > 0:
            legend_els.append(Line2D([0], [0], color="#2D93AD", ls=":", lw=1.5, label="Baseline Best"))
        ax.legend(handles=legend_els, loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_title(f"{title}\n{metrics}", fontsize=11, fontweight="bold", pad=10)
        ax.set_aspect("equal")
        _auto_lim(ax, past, gt_rel, baseline["best"])
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        f"LAformer Ablation: {scene_label}\n"
        f"{n_veh} vehicles + {n_ped} pedestrians + {n_lanes} lanes",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "six_panel_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_top3_and_ranking(
    mapping, neighbor_info, baseline, indiv_results, groups, out_dir, scene_label,
):
    """Top-3 individual ablations + ranking bar chart."""
    past = np.array(mapping["past_traj"])[:, :2]
    gt_rel = np.array(mapping["labels"])
    lanes = mapping.get("polygons", [])

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    for pi in range(min(3, len(indiv_results))):
        ax = axes[pi]
        r = indiv_results[pi]
        _draw_lanes(ax, lanes)
        for info in neighbor_info:
            t = info["traj"]
            if info["idx"] == r["idx"]:
                ax.plot(t[:, 0], t[:, 1], color="red", lw=3, alpha=0.9, zorder=6)
                ax.plot(t[-1, 0], t[-1, 1], "X", color="red", ms=14, zorder=7, mec="white", mew=1)
            else:
                c = VEH_COLOR if info["type"] == "Vehicle" else (PED_COLOR if info["type"] == "Pedestrian" else "gray")
                ax.plot(t[:, 0], t[:, 1], color=c, lw=0.8, alpha=0.25, zorder=2)
        _draw_ego_and_gt(ax, past, gt_rel)
        ax.plot(baseline["best"][:, 0], baseline["best"][:, 1], color="#2D93AD", lw=2.5, label="Baseline", zorder=10)
        ax.plot(r["pred"][:, 0], r["pred"][:, 1], "--", color="#E74C3C", lw=2.5,
                label=f'Without (shift={r["ade_shift"]:.3f}m)', zorder=10)
        ax.legend(loc="upper right", fontsize=8)
        short = r["cat"].split(".")[-1] if "." in r["cat"] else r["cat"]
        ax.set_title(f'Remove #{pi+1}: {short} ({r["dist"]:.1f}m)\nADE shift={r["ade_shift"]:.3f}m', fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        _auto_lim(ax, past, gt_rel, baseline["best"])
        ax.grid(True, alpha=0.2)

    ax = axes[3]
    top_n = indiv_results[:min(15, len(indiv_results))]
    max_shift = max(r["ade_shift"] for r in top_n) if top_n else 1
    labels = [f'{r["type"][:3]}\n{r["dist"]:.0f}m' for r in top_n]
    shifts = [r["ade_shift"] for r in top_n]
    colors_bar = [plt.cm.YlOrRd(s / (max_shift + 1e-8) * 0.8 + 0.1) for s in shifts]
    ax.barh(range(len(top_n)), shifts, color=colors_bar, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    if "no_agents" in groups:
        ax.axvline(groups["no_agents"]["ade_shift"], color="#333333", ls="--", lw=1.5, alpha=0.7,
                   label=f'All removed: {groups["no_agents"]["ade_shift"]:.3f}m')
    ax.set_xlabel("ADE Shift (m)", fontsize=11)
    ax.set_title("Neighbor Influence Ranking", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"Individual Neighbor Ablation: {scene_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "top3_and_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_temporal(
    mapping, neighbor_info, baseline, temporal_results, out_dir, scene_label,
):
    """Temporal occlusion trajectories + bar chart."""
    past = np.array(mapping["past_traj"])[:, :2]
    gt_rel = np.array(mapping["labels"])
    lanes = mapping.get("polygons", [])
    num_hist = len(past)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.2, 1]})

    _draw_lanes(ax1, lanes)
    _draw_agents(ax1, neighbor_info)
    ax1.plot(past[:, 0], past[:, 1], "--", color="#333333", lw=2.5, label="History", zorder=7)
    ax1.plot(past[-1, 0], past[-1, 1], "o", color="#333333", ms=10, zorder=8)
    ax1.plot(baseline["best"][:, 0], baseline["best"][:, 1], color="#2D93AD", lw=2.5, label="Baseline", zorder=10)

    cmap = plt.cm.plasma
    for r in temporal_results:
        c = cmap(r["recency"] / max(1, num_hist - 1))
        ax1.plot(r["pred"][:, 0], r["pred"][:, 1], color=c, lw=2.0, alpha=0.8, zorder=8,
                 label=f't-{r["recency"]} (shift={r["ade_shift"]:.3f}m)')
    ax1.plot(gt_rel[:, 0], gt_rel[:, 1], color="#E74C8B", lw=3.5, alpha=0.9, zorder=9,
             path_effects=[pe.withStroke(linewidth=5.5, foreground="white", alpha=0.4)])
    ax1.set_title("Temporal Occlusion: Perturbed Predictions", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_aspect("equal")
    _auto_lim(ax1, past, gt_rel, baseline["best"])
    ax1.grid(True, alpha=0.2)

    recencies = [r["recency"] for r in temporal_results]
    ade_shifts = [r["ade_shift"] for r in temporal_results]
    fde_shifts = [r["fde_shift"] for r in temporal_results]
    max_s = max(ade_shifts) if ade_shifts else 1
    x_t = np.arange(len(recencies))
    w = 0.35
    ax2.bar(x_t - w / 2, ade_shifts, w,
            color=[plt.cm.YlOrRd(s / (max_s + 1e-8) * 0.8 + 0.1) for s in ade_shifts],
            edgecolor="white", lw=1, label="ADE Shift")
    ax2.bar(x_t + w / 2, fde_shifts, w,
            color=[plt.cm.YlOrRd(f / (max(fde_shifts) + 1e-8) * 0.8 + 0.1) for f in fde_shifts],
            edgecolor="white", lw=1, alpha=0.6, label="FDE Shift")
    ax2.set_xlabel(f"History Step (t-0=current, t-{num_hist-1}=oldest)", fontsize=11)
    ax2.set_ylabel("Prediction Shift (m)", fontsize=11)
    ax2.set_xticks(x_t)
    ax2.set_xticklabels([f"t-{r}" for r in recencies])
    ax2.set_title("Temporal Importance", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Temporal Occlusion: {scene_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "temporal_occlusion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_group_summary(
    neighbor_info, indiv_results, groups, out_dir, scene_label,
):
    """Group ablation bars + individual scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    cats = []
    ade_vals = []
    fde_vals = []
    colors = []
    for key, c, label in [
        ("ego_only", "#222222", "Ego Only"),
        ("no_agents", "#555555", "All Neighbors"),
        ("no_lanes", "#4A90D9", "All Lanes"),
        ("no_vehs", VEH_COLOR, "All Vehicles"),
        ("no_peds", PED_COLOR, "All Pedestrians"),
    ]:
        if key in groups:
            cats.append(label)
            ade_vals.append(groups[key]["ade_shift"])
            fde_vals.append(groups[key]["fde_shift"])
            colors.append(c)

    x_pos = np.arange(len(cats))
    w = 0.35
    bars1 = ax1.bar(x_pos - w / 2, ade_vals, w, color=colors, edgecolor="white", lw=1, label="ADE Shift")
    bars2 = ax1.bar(x_pos + w / 2, fde_vals, w, color=colors, edgecolor="white", lw=1, alpha=0.6, label="FDE Shift")
    for b, v in zip(bars1, ade_vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cats, fontsize=11)
    ax1.set_ylabel("Shift (m)", fontsize=12)
    ax1.set_title("Group Ablation: Prediction Shift", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    for ntype, marker, color, label in [
        ("Vehicle", "s", VEH_COLOR, "Vehicles"),
        ("Pedestrian", "o", PED_COLOR, "Pedestrians"),
        ("Other", "^", "gray", "Other"),
    ]:
        xs = [r["dist"] for r in indiv_results if r["type"] == ntype]
        ys = [r["ade_shift"] for r in indiv_results if r["type"] == ntype]
        if xs:
            ax2.scatter(xs, ys, c=color, s=60, alpha=0.7, label=f"{label} ({len(xs)})",
                        edgecolors="white", lw=0.5, marker=marker)
    if indiv_results:
        top = indiv_results[0]
        ax2.annotate(f'{top["type"]}\n{top["dist"]:.0f}m, shift={top["ade_shift"]:.2f}m',
                     (top["dist"], top["ade_shift"]), textcoords="offset points", xytext=(10, 10),
                     fontsize=9, fontweight="bold", color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    ax2.set_xlabel("Distance from Ego (m)", fontsize=12)
    ax2.set_ylabel("ADE Shift (m)", fontsize=12)
    ax2.set_title("Individual Influence vs Distance", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Group Ablation Summary: {scene_label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "group_ablation_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_sample(samples, nusc, scene=None, n_agents=None, sample_idx=None):
    """Resolve a sample from the cache by scene name, agent count, or index."""
    if sample_idx is not None:
        return sample_idx, samples[sample_idx]

    candidates = []
    for i, s in enumerate(samples):
        if scene:
            sample_rec = nusc.get("sample", s["sample_token"])
            scene_rec = nusc.get("scene", sample_rec["scene_token"])
            if scene_rec["name"] != scene:
                continue
        na = len(s.get("trajs", []))
        if n_agents and na != n_agents:
            continue
        candidates.append((i, s, na))

    if not candidates:
        raise ValueError(f"No sample found for scene={scene}, n_agents={n_agents}")

    candidates.sort(key=lambda x: x[2], reverse=True)
    idx, mapping, na = candidates[0]
    logger.info("Selected sample idx=%d (%d agents)", idx, na)
    return idx, mapping


def main():
    parser = argparse.ArgumentParser(description="LAformer Explainability Analysis")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Direct sample index in the cache")
    parser.add_argument("--scene", type=str, default=None,
                        help="nuScenes scene name (e.g. scene-0103)")
    parser.add_argument("--n_agents", type=int, default=None,
                        help="Filter by exact number of agents")
    parser.add_argument("--cache_path", type=str,
                        default=str(PROJ_ROOT / "data" / "mini_val_cache" / "eval.ex_list"),
                        help="Path to preprocessed LAformer cache")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LAformer checkpoint")
    parser.add_argument("--nusc_datadir", type=str,
                        default=str(PROJ_ROOT / "nuscenes_mini"),
                        help="Path to nuScenes dataset root")
    parser.add_argument("--nusc_version", type=str, default="v1.0-mini")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJ_ROOT / "results" / "explainability"),
                        help="Output directory for figures")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_temporal", action="store_true",
                        help="Skip temporal occlusion analysis")
    args = parser.parse_args()

    if args.sample_idx is None and args.scene is None:
        parser.error("Provide either --sample_idx or --scene")

    device = torch.device(args.device)

    logger.info("Loading LAformer model...")
    model, _ = load_model(args.checkpoint, args.device)

    logger.info("Loading samples from %s", args.cache_path)
    samples = load_samples(args.cache_path)

    logger.info("Loading nuScenes for GT annotations...")
    from nuscenes import NuScenes
    nusc = NuScenes(args.nusc_version, dataroot=args.nusc_datadir, verbose=False)

    idx, mapping = find_sample(samples, nusc, args.scene, args.n_agents, args.sample_idx)
    sample_rec = nusc.get("sample", mapping["sample_token"])
    scene_rec = nusc.get("scene", sample_rec["scene_token"])
    scene_name = scene_rec["name"]
    n_total = len(mapping["trajs"])

    out_dir = os.path.join(args.output_dir, f"{scene_name}_idx{idx}")
    os.makedirs(out_dir, exist_ok=True)
    scene_label = f"{scene_name} idx={idx} ({n_total} agents)"
    logger.info("Analyzing: %s", scene_label)

    neighbor_info = classify_neighbors(mapping, nusc)
    n_veh = sum(1 for n in neighbor_info if n["type"] == "Vehicle")
    n_ped = sum(1 for n in neighbor_info if n["type"] == "Pedestrian")
    n_other = sum(1 for n in neighbor_info if n["type"] == "Other")
    mislabeled = [n for n in neighbor_info if n["mislabeled"]]
    logger.info("Agents: %d vehicles, %d pedestrians, %d other", n_veh, n_ped, n_other)
    if mislabeled:
        logger.warning("%d agents mislabeled in LAformer encoding:", len(mislabeled))
        for n in mislabeled:
            logger.warning("  idx=%d enc=%s actual=%s dist=%.1fm", n["idx"], n["encoded"], n["cat"], n["dist"])

    gt_rel = np.array(mapping["labels"])
    preds_base, scores_base = predict(model, mapping, device)
    best_base = preds_base[np.argmax(scores_base)]
    base_metrics = compute_metrics(best_base, gt_rel)
    logger.info("Baseline: ADE=%.2fm, FDE=%.2fm", base_metrics["ade"], base_metrics["fde"])

    baseline = {"best": best_base, "all_modes": preds_base, "scores": scores_base}

    logger.info("Running individual neighbor ablation (%d agents)...", n_total)
    indiv_results = run_neighbor_ablation(model, mapping, device, neighbor_info, best_base)
    logger.info("Top 5 influential:")
    for i, r in enumerate(indiv_results[:5]):
        logger.info("  %d. %s at %.1fm -> ADE shift=%.3fm", i + 1, r["cat"], r["dist"], r["ade_shift"])

    logger.info("Running group ablations...")
    groups = run_group_ablation(model, mapping, device, neighbor_info, best_base, gt_rel)
    for key, g in groups.items():
        logger.info("  %s: ADE shift=%.3fm, FDE shift=%.3fm", key, g["ade_shift"], g["fde_shift"])

    temporal_results = None
    if not args.skip_temporal:
        logger.info("Running temporal occlusion...")
        temporal_results = run_temporal_occlusion(model, mapping, device, best_base)
        most_imp = max(temporal_results, key=lambda x: x["ade_shift"])
        logger.info("Most important timestep: t-%d (shift=%.3fm)", most_imp["recency"], most_imp["ade_shift"])

    logger.info("Generating figures...")
    plot_six_panel(mapping, neighbor_info, baseline, groups, out_dir, scene_label)
    plot_top3_and_ranking(mapping, neighbor_info, baseline, indiv_results, groups, out_dir, scene_label)
    plot_group_summary(neighbor_info, indiv_results, groups, out_dir, scene_label)
    if temporal_results:
        plot_temporal(mapping, neighbor_info, baseline, temporal_results, out_dir, scene_label)

    summary = {
        "scene": scene_name,
        "sample_idx": idx,
        "n_agents": n_total,
        "n_vehicles": n_veh,
        "n_pedestrians": n_ped,
        "n_other": n_other,
        "n_mislabeled": len(mislabeled),
        "baseline_ade": base_metrics["ade"],
        "baseline_fde": base_metrics["fde"],
        "group_ablation": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in groups.items()},
        "top5_individual": [
            {"type": r["type"], "cat": r["cat"], "dist": r["dist"],
             "ade_shift": r["ade_shift"], "fde_shift": r["fde_shift"]}
            for r in indiv_results[:5]
        ],
    }
    if temporal_results:
        summary["temporal"] = [
            {"t": r["t"], "recency": r["recency"],
             "ade_shift": r["ade_shift"], "fde_shift": r["fde_shift"]}
            for r in temporal_results
        ]
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", json_path)

    logger.info("All done! Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
