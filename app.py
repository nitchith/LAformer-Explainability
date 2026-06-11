"""
LAformer Interactive Explainability Demo

Gradio web app for interactively toggling agents and lanes to see
how LAformer's trajectory predictions change.

Usage:
    conda activate laformer-explain
    python app.py
"""

import copy
import logging
import sys
from io import BytesIO
from pathlib import Path

import gradio as gr

# Patch gradio_client bug: additionalProperties can be a bool in JSON Schema,
# but gradio_client assumes it's always a dict.
import gradio_client.utils as _gc_utils
_orig_get_type = _gc_utils.get_type
_orig_jspy = _gc_utils._json_schema_to_python_type
def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "any"
    return _orig_get_type(schema)
def _patched_jspy(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig_jspy(schema, defs)
_gc_utils.get_type = _patched_get_type
_gc_utils._json_schema_to_python_type = _patched_jspy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "laformer"))
sys.path.insert(0, str(PROJ_ROOT))

from explainability.explain_utils import (
    load_model,
    load_samples,
    predict,
    classify_neighbors,
    remove_neighbors,
    remove_lanes,
    remove_specific_lanes,
    compute_metrics,
    compute_shift,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODE_COLORS = ["#2D93AD", "#27AE60", "#F39C12", "#9B59B6", "#E74C3C"]
VEH_COLOR = "#6B8E23"
PED_COLOR = "#D35400"
OTHER_COLOR = "#7F8C8D"

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL = None
DEVICE = "cpu"
CACHES = {}
NUSC = None
CURRENT_SAMPLES = []
CURRENT_NEIGHBOR_INFO = []
CURRENT_MAPPING = None
CURRENT_BASELINE = None


def startup():
    global MODEL, NUSC, CACHES
    logger.info("Loading LAformer model...")
    MODEL, _ = load_model(device=DEVICE)

    val_path = PROJ_ROOT / "data" / "mini_val_cache" / "eval.ex_list"
    train_path = PROJ_ROOT / "data" / "mini_train_cache" / "ex_list"

    if val_path.exists():
        CACHES["val"] = load_samples(str(val_path))
    if train_path.exists():
        CACHES["train"] = load_samples(str(train_path))

    nusc_path = PROJ_ROOT / "nuscenes_mini"
    if nusc_path.exists():
        logger.info("Loading nuScenes...")
        from nuscenes import NuScenes
        NUSC = NuScenes("v1.0-mini", dataroot=str(nusc_path), verbose=False)
    else:
        logger.warning("nuscenes_mini/ not found -- agent types will use LAformer encoding")

    logger.info("Startup complete. Caches: %s", list(CACHES.keys()))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

LANE_LABEL_RADIUS = 25.0

def render_scene(mapping, neighbor_info, preds, scores, gt_rel,
                 removed_indices=None, lanes_removed=False, title="",
                 baseline_best=None, removed_lane_indices=None,
                 zoom_level=0, show_lane_labels=False):
    """Render a scene and return a PIL Image."""
    removed_indices = removed_indices or set()
    removed_lane_indices = removed_lane_indices or set()
    past = np.array(mapping["past_traj"])[:, :2]
    lanes = mapping.get("polygons", [])
    ego_pos = past[-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for li, lane in enumerate(lanes):
        l = np.array(lane)
        mid = l[len(l) // 2]
        dist_to_ego = np.linalg.norm(mid - ego_pos)
        should_label = show_lane_labels or (li in removed_lane_indices and dist_to_ego < LANE_LABEL_RADIUS)

        if lanes_removed or li in removed_lane_indices:
            ax.plot(l[:, 0], l[:, 1], color="#CCCCCC", lw=0.5, alpha=0.15, ls=":", zorder=1)
            if should_label and li in removed_lane_indices:
                ax.annotate(f"L{li}", xy=(mid[0], mid[1]),
                            fontsize=6, color="red", alpha=0.4, fontweight="bold",
                            ha="center", va="center", zorder=2)
        else:
            ax.plot(l[:, 0], l[:, 1], color="#CCCCCC", lw=0.8, alpha=0.6, zorder=1)
            if should_label:
                ax.annotate(f"L{li}", xy=(mid[0], mid[1]),
                            fontsize=6, color="#999999", fontweight="bold",
                            ha="center", va="center", zorder=2,
                            path_effects=[pe.withStroke(linewidth=1.5, foreground="white", alpha=0.9)])

    for info in neighbor_info:
        t = info["traj"]
        is_removed = info["idx"] in removed_indices
        if info["type"] == "Vehicle":
            color, mk = VEH_COLOR, "s"
            type_short = "V"
        elif info["type"] == "Pedestrian":
            color, mk = PED_COLOR, "o"
            type_short = "P"
        else:
            color, mk = OTHER_COLOR, "^"
            type_short = "O"

        label_text = f"#{info['idx']} {type_short}"

        if is_removed:
            ax.plot(t[:, 0], t[:, 1], color=color, lw=0.8, alpha=0.12, ls=":", zorder=2)
            ax.plot(t[-1, 0], t[-1, 1], "x", color="red", ms=7, alpha=0.5, zorder=3, markeredgewidth=2)
            ax.annotate(label_text, xy=(t[-1, 0], t[-1, 1]),
                        fontsize=7, color="red", alpha=0.4, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=5)
        else:
            alpha = 0.7 if info["type"] == "Pedestrian" else 0.5
            ax.plot(t[:, 0], t[:, 1], color=color, lw=1.5, alpha=alpha, zorder=3)
            ax.plot(t[-1, 0], t[-1, 1], mk, color=color, ms=5, alpha=0.6, zorder=4,
                    mec="white", mew=0.3)
            ax.annotate(label_text, xy=(t[-1, 0], t[-1, 1]),
                        fontsize=7.5, color=color, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=5,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])

    ax.plot(past[:, 0], past[:, 1], "--", color="#333333", lw=2.5, zorder=7)
    ax.plot(past[-1, 0], past[-1, 1], "o", color="#333333", ms=10, zorder=8)

    ax.plot(gt_rel[:, 0], gt_rel[:, 1], color="#E74C8B", lw=3.5, alpha=0.9, zorder=9,
            path_effects=[pe.withStroke(linewidth=5.5, foreground="white", alpha=0.4)])
    ax.plot(gt_rel[-1, 0], gt_rel[-1, 1], "*", color="#E74C8B", ms=14, zorder=11,
            mec="white", mew=0.8)

    if baseline_best is not None:
        ax.plot(baseline_best[:, 0], baseline_best[:, 1], ":", color="#2D93AD",
                lw=1.8, alpha=0.5, zorder=7, label="Baseline best")

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

    traj_pts = np.concatenate([past, gt_rel, preds[sorted_m[0]]])
    agent_pts = np.array([info["traj"][-1] for info in neighbor_info]) if neighbor_info else np.empty((0, 2))

    # zoom_level: 0 = default (prediction-focused), positive = zoom out, negative = zoom in
    if zoom_level <= -2:
        pad = 3
        all_pts = traj_pts
    elif zoom_level == -1:
        pad = 5
        all_pts = traj_pts
    elif zoom_level == 0:
        pad = 8
        all_pts = traj_pts
    elif zoom_level == 1:
        pad = 8
        all_pts = np.concatenate([traj_pts, agent_pts]) if len(agent_pts) else traj_pts
    else:
        lane_pts = []
        for lane in lanes:
            l = np.array(lane)
            lane_pts.append(l[len(l) // 2])
        if lane_pts:
            all_pts = np.concatenate([traj_pts, agent_pts, np.array(lane_pts)])
        elif len(agent_pts):
            all_pts = np.concatenate([traj_pts, agent_pts])
        else:
            all_pts = traj_pts
        pad = 10

    xmn, xmx = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymn, ymx = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    rng = max(xmx - xmn, ymx - ymn)
    mx, my = (xmn + xmx) / 2, (ymn + ymx) / 2
    ax.set_xlim(mx - rng / 2, mx + rng / 2)
    ax.set_ylim(my - rng / 2, my + rng / 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=2.5, ls="--", label="Ego Past"),
        Line2D([0], [0], color="#E74C8B", lw=3.5, label="Ground Truth"),
        Line2D([0], [0], color=MODE_COLORS[0], lw=2.5, label="Best Prediction"),
        Line2D([0], [0], color=MODE_COLORS[1], lw=1.5, alpha=0.45, label="Other Modes"),
        Line2D([0], [0], color=VEH_COLOR, lw=1.5, marker="s", ms=5, label="Vehicle"),
        Line2D([0], [0], color=PED_COLOR, lw=1.5, marker="o", ms=5, label="Pedestrian"),
        Line2D([0], [0], color="#CCCCCC", lw=0.8, label="Lanes"),
    ]
    if baseline_best is not None:
        legend_handles.append(
            Line2D([0], [0], color="#2D93AD", lw=1.8, ls=":", alpha=0.5, label="Baseline Best"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7.5,
              framealpha=0.85, edgecolor="#CCCCCC", fancybox=True)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

SAMPLE_METRICS_CACHE = {}

def _precompute_metrics(split):
    """Compute FDE for all samples in a split (cached)."""
    if split in SAMPLE_METRICS_CACHE:
        return SAMPLE_METRICS_CACHE[split]
    samples = CACHES.get(split, [])
    metrics_list = []
    for m in samples:
        gt_rel = np.array(m["labels"])
        preds, scores = predict(MODEL, m, DEVICE)
        best = preds[np.argmax(scores)]
        met = compute_metrics(best, gt_rel)
        metrics_list.append(met)
    SAMPLE_METRICS_CACHE[split] = metrics_list
    logger.info("Precomputed metrics for %s: %d samples", split, len(metrics_list))
    return metrics_list


def get_sample_choices(split):
    samples = CACHES.get(split, [])
    metrics_list = _precompute_metrics(split)
    choices = []
    for i, m in enumerate(samples):
        n_agents = len(m.get("trajs", []))
        scene = ""
        if NUSC and "sample_token" in m:
            try:
                rec = NUSC.get("sample", m["sample_token"])
                sc = NUSC.get("scene", rec["scene_token"])
                scene = sc["name"]
            except Exception:
                scene = "?"
        fde = metrics_list[i]["fde"] if i < len(metrics_list) else 0.0
        choices.append(f"[{i}] {scene} | {n_agents} agents | FDE={fde:.1f}m")
    return choices


def on_sample_select(split, sample_choice, zoom_level=0, show_lane_labels=False):
    """Called when a sample is selected. Returns baseline image + agent checkboxes."""
    global CURRENT_SAMPLES, CURRENT_NEIGHBOR_INFO, CURRENT_MAPPING, CURRENT_BASELINE

    if not sample_choice:
        return None, [], ""

    idx = int(sample_choice.split("]")[0].strip("["))
    samples = CACHES.get(split, [])
    mapping = samples[idx]
    CURRENT_MAPPING = mapping
    CURRENT_SAMPLES = samples

    neighbor_info = classify_neighbors(mapping, NUSC)
    CURRENT_NEIGHBOR_INFO = neighbor_info

    gt_rel = np.array(mapping["labels"])
    preds, scores = predict(MODEL, mapping, DEVICE)
    best = preds[np.argmax(scores)]
    metrics = compute_metrics(best, gt_rel)
    CURRENT_BASELINE = {"best": best, "preds": preds, "scores": scores}

    n_veh = sum(1 for n in neighbor_info if n["type"] == "Vehicle")
    n_ped = sum(1 for n in neighbor_info if n["type"] == "Pedestrian")
    n_lanes = len(mapping["polyline_spans"]) - mapping["map_start_polyline_idx"]

    scene_name = ""
    if NUSC:
        try:
            rec = NUSC.get("sample", mapping["sample_token"])
            sc = NUSC.get("scene", rec["scene_token"])
            scene_name = sc["name"]
        except Exception:
            pass

    title = (f"Baseline: {scene_name} idx={idx}\n"
             f"{n_veh}V + {n_ped}P + {n_lanes} lanes | "
             f"ADE={metrics['ade']:.2f}m  FDE={metrics['fde']:.2f}m")
    img = render_scene(mapping, neighbor_info, preds, scores, gt_rel,
                       title=title, zoom_level=int(zoom_level),
                       show_lane_labels=show_lane_labels)

    agent_choices = []
    for info in neighbor_info:
        short_cat = info["cat"].split(".")[-1] if "." in info["cat"] else info["cat"]
        label = f"#{info['idx']} {info['type']} ({short_cat}) @ {info['dist']:.1f}m"
        agent_choices.append(label)

    info_text = (f"**Scene:** {scene_name} | **Sample idx:** {idx}\n\n"
                 f"**Agents:** {n_veh} vehicles, {n_ped} pedestrians | "
                 f"**Lanes:** {n_lanes}\n\n"
                 f"**Baseline ADE:** {metrics['ade']:.2f}m | "
                 f"**Baseline FDE:** {metrics['fde']:.2f}m")

    return img, agent_choices, info_text


def _parse_lane_indices(text: str) -> set:
    """Parse a string like '0,1,5-10,20' into a set of ints."""
    indices = set()
    if not text or not text.strip():
        return indices
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                indices.update(range(int(lo.strip()), int(hi.strip()) + 1))
            except ValueError:
                continue
        else:
            try:
                indices.add(int(part))
            except ValueError:
                continue
    return indices


def on_ablate(remove_peds, remove_vehs, remove_lanes_flag, agents_to_remove, lane_indices_text, zoom_level=0, show_lane_labels=False):
    """Called when toggles change. Returns ablated image + metrics."""
    if CURRENT_MAPPING is None or CURRENT_BASELINE is None:
        return None, "Select a sample first."

    mapping = CURRENT_MAPPING
    neighbor_info = CURRENT_NEIGHBOR_INFO
    gt_rel = np.array(mapping["labels"])
    baseline_best = CURRENT_BASELINE["best"]

    indices_to_remove = set()

    if remove_peds:
        for n in neighbor_info:
            if n["type"] == "Pedestrian":
                indices_to_remove.add(n["idx"])
    if remove_vehs:
        for n in neighbor_info:
            if n["type"] == "Vehicle":
                indices_to_remove.add(n["idx"])

    if agents_to_remove:
        for label in agents_to_remove:
            idx_str = label.split("#")[1].split(" ")[0]
            indices_to_remove.add(int(idx_str))

    lane_indices_to_remove = _parse_lane_indices(lane_indices_text)

    m_abl = mapping
    if indices_to_remove:
        m_abl = remove_neighbors(m_abl, sorted(indices_to_remove))
    if remove_lanes_flag:
        m_abl = remove_lanes(m_abl)
    elif lane_indices_to_remove:
        m_abl = remove_specific_lanes(m_abl, sorted(lane_indices_to_remove))

    preds, scores = predict(MODEL, m_abl, DEVICE)
    best = preds[np.argmax(scores)]
    metrics = compute_metrics(best, gt_rel)
    shift = compute_shift(baseline_best, best)

    n_rm = len(indices_to_remove)
    parts = []
    if remove_peds:
        parts.append("No Peds")
    if remove_vehs:
        parts.append("No Vehs")
    if remove_lanes_flag:
        parts.append("No Lanes")
    elif lane_indices_to_remove:
        parts.append(f"{len(lane_indices_to_remove)} lanes removed")
    extra_agents = [a for a in (agents_to_remove or [])
                    if not (remove_peds and "Pedestrian" in a) and not (remove_vehs and "Vehicle" in a)]
    if extra_agents and not (remove_peds and remove_vehs):
        parts.append(f"+{len(extra_agents)} individual agents")
    ablation_desc = ", ".join(parts) if parts else "No changes"

    title = (f"Ablated: {ablation_desc}\n"
             f"{n_rm} agents removed | "
             f"ADE={metrics['ade']:.2f}m  FDE={metrics['fde']:.2f}m")

    img = render_scene(
        CURRENT_MAPPING, neighbor_info, preds, scores, gt_rel,
        removed_indices=indices_to_remove,
        lanes_removed=remove_lanes_flag,
        title=title,
        baseline_best=baseline_best,
        removed_lane_indices=lane_indices_to_remove if not remove_lanes_flag else None,
        zoom_level=int(zoom_level),
        show_lane_labels=show_lane_labels,
    )

    base_m = compute_metrics(baseline_best, gt_rel)
    n_lanes_rm = "all" if remove_lanes_flag else str(len(lane_indices_to_remove))
    result_text = (
        f"### Ablation: {ablation_desc}\n\n"
        f"| Metric | Baseline | Ablated | Shift |\n"
        f"|--------|----------|---------|-------|\n"
        f"| ADE | {base_m['ade']:.3f}m | {metrics['ade']:.3f}m | "
        f"**{shift['ade_shift']:.3f}m** |\n"
        f"| FDE | {base_m['fde']:.3f}m | {metrics['fde']:.3f}m | "
        f"**{shift['fde_shift']:.3f}m** |\n\n"
        f"Removed **{n_rm}** agents + **{n_lanes_rm}** lanes"
    )

    return img, result_text


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="LAformer Explainability",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# LAformer Interactive Explainability Demo\n"
                     "Select a sample, then toggle agents and lanes to see how "
                     "the prediction changes in real-time.")

        with gr.Row():
            split_dd = gr.Dropdown(
                choices=list(CACHES.keys()),
                value=list(CACHES.keys())[0] if CACHES else None,
                label="Dataset Split",
                scale=1,
            )
            sample_dd = gr.Dropdown(
                choices=[],
                label="Select Sample",
                scale=3,
            )
            load_btn = gr.Button("Load Sample", variant="primary", scale=1)

        info_md = gr.Markdown("")

        with gr.Row():
            zoom_slider = gr.Slider(
                minimum=-2, maximum=2, step=1, value=0,
                label="Zoom: -2 (close-up) ← 0 (default) → +2 (full scene)",
                scale=3,
            )
            show_lane_cb = gr.Checkbox(label="Show lane numbers", value=False, scale=1)

        with gr.Row(equal_height=True):
            baseline_img = gr.Image(label="Baseline Prediction", type="pil")
            ablated_img = gr.Image(label="Ablated Prediction", type="pil")

        gr.Markdown("### Ablation Controls")
        with gr.Row():
            rm_peds = gr.Checkbox(label="Remove all Pedestrians", value=False)
            rm_vehs = gr.Checkbox(label="Remove all Vehicles", value=False)
            rm_lanes = gr.Checkbox(label="Remove all Lanes", value=False)

        agents_cb = gr.CheckboxGroup(
            choices=[], label="Remove Individual Agents (select to remove)",
        )

        lane_indices_txt = gr.Textbox(
            label="Remove Specific Lanes (enter indices, e.g. 0,3,5-10)",
            placeholder="e.g. 0,1,5-10,20",
            value="",
        )

        ablate_btn = gr.Button("Run Ablation", variant="primary")
        result_md = gr.Markdown("")

        def update_samples(split):
            choices = get_sample_choices(split)
            return gr.Dropdown(choices=choices, value=None, label="Select Sample")

        split_dd.change(fn=update_samples, inputs=[split_dd], outputs=[sample_dd])

        def load_sample(split, sample_choice, zoom, show_lanes):
            img, agent_choices, info_text = on_sample_select(split, sample_choice, zoom, show_lanes)
            return (
                img,
                None,
                gr.CheckboxGroup(choices=agent_choices, value=[], label="Remove Individual Agents (select to remove)"),
                info_text,
                "",
                False,
                False,
                False,
                "",
            )

        load_btn.click(
            fn=load_sample,
            inputs=[split_dd, sample_dd, zoom_slider, show_lane_cb],
            outputs=[baseline_img, ablated_img, agents_cb, info_md, result_md,
                     rm_peds, rm_vehs, rm_lanes, lane_indices_txt],
        )

        def refresh_baseline(split, sample_choice, zoom, show_lanes):
            """Re-render baseline with new zoom/labels without re-running inference."""
            if CURRENT_MAPPING is None or CURRENT_BASELINE is None:
                return None
            mapping = CURRENT_MAPPING
            gt_rel = np.array(mapping["labels"])
            preds = CURRENT_BASELINE["preds"]
            scores = CURRENT_BASELINE["scores"]
            best = preds[np.argmax(scores)]
            metrics = compute_metrics(best, gt_rel)
            n_veh = sum(1 for n in CURRENT_NEIGHBOR_INFO if n["type"] == "Vehicle")
            n_ped = sum(1 for n in CURRENT_NEIGHBOR_INFO if n["type"] == "Pedestrian")
            n_lanes = len(mapping["polyline_spans"]) - mapping["map_start_polyline_idx"]
            scene_name = ""
            if NUSC and "sample_token" in mapping:
                try:
                    rec = NUSC.get("sample", mapping["sample_token"])
                    sc = NUSC.get("scene", rec["scene_token"])
                    scene_name = sc["name"]
                except Exception:
                    pass
            title = (f"Baseline: {scene_name}\n"
                     f"{n_veh}V + {n_ped}P + {n_lanes} lanes | "
                     f"ADE={metrics['ade']:.2f}m  FDE={metrics['fde']:.2f}m")
            return render_scene(mapping, CURRENT_NEIGHBOR_INFO, preds, scores,
                                gt_rel, title=title, zoom_level=int(zoom),
                                show_lane_labels=show_lanes)

        zoom_slider.change(
            fn=refresh_baseline,
            inputs=[split_dd, sample_dd, zoom_slider, show_lane_cb],
            outputs=[baseline_img],
        )
        show_lane_cb.change(
            fn=refresh_baseline,
            inputs=[split_dd, sample_dd, zoom_slider, show_lane_cb],
            outputs=[baseline_img],
        )

        ablate_btn.click(
            fn=on_ablate,
            inputs=[rm_peds, rm_vehs, rm_lanes, agents_cb, lane_indices_txt, zoom_slider, show_lane_cb],
            outputs=[ablated_img, result_md],
        )

        default_split = list(CACHES.keys())[0] if CACHES else "val"
        demo.load(
            fn=lambda: gr.Dropdown(
                choices=get_sample_choices(default_split),
                value=None, label="Select Sample"),
            outputs=[sample_dd],
        )

    return demo


if __name__ == "__main__":
    startup()
    demo = build_ui()
    demo.launch()
