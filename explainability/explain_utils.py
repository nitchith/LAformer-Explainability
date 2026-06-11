"""
Utility functions for LAformer explainability analysis.

Handles model loading, data loading, prediction, neighbor classification
(with nuScenes GT cross-referencing), and perturbation helpers.
"""

import copy
import logging
import pickle
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[1]
LAFORMER_SRC = PROJ_ROOT / "laformer"


def _init_laformer_args():
    """Build the argparse Namespace that LAformer expects."""
    import sys
    sys.path.insert(0, str(LAFORMER_SRC))
    import argparse
    from utils_files import config

    parser = argparse.ArgumentParser()
    config.add_argument(parser)
    args = parser.parse_args([
        "--future_frame_num", "12",
        "--hidden_size", "64",
        "--topk", "2",
        "--eval_batch_size", "1",
        "--do_eval",
        "--output_dir", str(PROJ_ROOT / "checkpoints" / "nuScene_k5"),
        "--other_params",
        "semantic_lane", "direction", "step_lane_score",
        "enhance_global_graph", "point_level-4-3",
        "stage_two", "nuscenes", "nuscenes_mode_num=5",
    ])
    return args


def load_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, object]:
    """
    Load pretrained LAformer model.

    Returns (model, args).
    """
    import sys
    sys.path.insert(0, str(LAFORMER_SRC))
    from utils_files import utils
    from model_main import ModelMain

    args = _init_laformer_args()
    utils.init(args, logger)

    model = ModelMain(args)

    if checkpoint_path is None:
        checkpoint_path = str(
            PROJ_ROOT / "checkpoints" / "nuScene_k5" / "model.50.bin"
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("LAformer model loaded from %s", checkpoint_path)
    return model, args


def load_samples(cache_path: str) -> List[dict]:
    """Load preprocessed LAformer samples from a pickle cache."""
    with open(cache_path, "rb") as f:
        raw = pickle.load(f)
    samples = [pickle.loads(zlib.decompress(c)) for c in raw]
    logger.info("Loaded %d samples from %s", len(samples), cache_path)
    return samples


def abs_to_rel(points_abs: np.ndarray, cx: float, cy: float, angle: float) -> np.ndarray:
    """Convert absolute coordinates to ego-relative frame."""
    import sys
    sys.path.insert(0, str(LAFORMER_SRC))
    from utils_files.utils import rotate
    return np.array([rotate(p[0] - cx, p[1] - cy, angle) for p in points_abs])


def predict(
    model: torch.nn.Module,
    mapping: dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LAformer inference on a single mapping.

    Returns:
        preds_rel: (K, T, 2) predictions in relative coordinates
        scores: (K,) mode probabilities
    """
    cx, cy, angle = mapping["cent_x"], mapping["cent_y"], mapping["angle"]
    with torch.no_grad():
        pred_traj, pred_score, _ = model([mapping], device)

    pred_abs = pred_traj[0]
    scores = pred_score[0]
    preds_rel = np.zeros_like(pred_abs)
    for k in range(pred_abs.shape[0]):
        preds_rel[k] = abs_to_rel(pred_abs[k], cx, cy, angle)
    return preds_rel, scores


def classify_neighbors(
    mapping: dict,
    nusc=None,
) -> List[dict]:
    """
    Classify each neighbor agent using nuScenes GT annotations when available,
    falling back to LAformer's matrix encoding otherwise.

    Returns list of dicts with keys: idx, type, cat, dist, traj, encoded, mislabeled.
    """
    import sys
    sys.path.insert(0, str(LAFORMER_SRC))

    agent_trajs = mapping.get("trajs", [])
    cx, cy, angle = mapping["cent_x"], mapping["cent_y"], mapping["angle"]

    ann_positions = None
    if nusc is not None and "sample_token" in mapping:
        try:
            from utils_files.utils import rotate
            sample = nusc.get("sample", mapping["sample_token"])
            ann_positions = []
            for tok in sample["anns"]:
                ann = nusc.get("sample_annotation", tok)
                rx, ry = rotate(
                    ann["translation"][0] - cx,
                    ann["translation"][1] - cy,
                    angle,
                )
                ann_positions.append((ann["category_name"], rx, ry))
        except Exception:
            ann_positions = None

    results = []
    for ni in range(len(agent_trajs)):
        traj = np.array(agent_trajs[ni])[:, :2]
        dist = float(np.linalg.norm(traj[-1]))

        span = mapping["polyline_spans"][ni + 1]
        row = mapping["matrix"][span.start]
        enc = "VEH" if row[6] == 1.0 else "PED"

        if ann_positions is not None:
            best_cat, best_d = "unknown", float("inf")
            for cat, rx, ry in ann_positions:
                md = np.sqrt((traj[-1, 0] - rx) ** 2 + (traj[-1, 1] - ry) ** 2)
                if md < best_d:
                    best_d = md
                    best_cat = cat
            is_ped = best_cat.startswith("human")
            is_veh = best_cat.startswith("vehicle")
            ntype = "Pedestrian" if is_ped else ("Vehicle" if is_veh else "Other")
        else:
            best_cat = "vehicle" if enc == "VEH" else "human.pedestrian"
            ntype = "Vehicle" if enc == "VEH" else "Pedestrian"

        mislabeled = (
            (ntype == "Pedestrian" and enc == "VEH")
            or (ntype == "Vehicle" and enc == "PED")
        )

        results.append({
            "idx": ni,
            "type": ntype,
            "cat": best_cat,
            "dist": dist,
            "traj": traj,
            "encoded": enc,
            "mislabeled": mislabeled,
        })

    return results


def remove_neighbors(
    mapping: dict,
    indices_to_remove: List[int],
) -> dict:
    """
    Return a deep copy of *mapping* with the specified neighbor polylines removed.
    """
    m = copy.deepcopy(mapping)
    for n_idx in sorted(indices_to_remove, reverse=True):
        pi = n_idx + 1
        sp = m["polyline_spans"][pi]
        rlen = sp.stop - sp.start

        new_spans = []
        for i, s in enumerate(m["polyline_spans"]):
            if i == pi:
                continue
            if i < pi:
                new_spans.append(s)
            else:
                new_spans.append(slice(s.start - rlen, s.stop - rlen))

        mask = np.ones(len(m["matrix"]), dtype=bool)
        mask[sp.start:sp.stop] = False
        m["matrix"] = m["matrix"][mask]
        m["polyline_spans"] = new_spans
        m["map_start_polyline_idx"] -= 1

        t = list(m["trajs"])
        del t[n_idx]
        m["trajs"] = t
    return m


def remove_lanes(
    mapping: dict,
) -> dict:
    """
    Return a deep copy of *mapping* with all lane features zeroed out.
    We zero the matrix rows rather than removing polylines because
    LAformer's encoder expects at least one lane polyline to exist.
    """
    m = copy.deepcopy(mapping)
    map_start = m["map_start_polyline_idx"]
    for sp in m["polyline_spans"][map_start:]:
        m["matrix"][sp.start:sp.stop, :] = 0.0
    return m


def remove_lanes_and_neighbors(
    mapping: dict,
) -> dict:
    """Zero out all lanes AND remove all neighbor agents, keeping only ego."""
    all_agent_idx = list(range(len(mapping["trajs"])))
    m = remove_neighbors(mapping, all_agent_idx) if all_agent_idx else copy.deepcopy(mapping)
    map_start = m["map_start_polyline_idx"]
    for sp in m["polyline_spans"][map_start:]:
        m["matrix"][sp.start:sp.stop, :] = 0.0
    return m


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """Compute ADE / FDE between a single-mode prediction and ground truth."""
    errors = np.linalg.norm(pred - gt, axis=-1)
    return {"ade": float(np.mean(errors)), "fde": float(errors[-1])}


def compute_shift(
    pred_baseline: np.ndarray,
    pred_perturbed: np.ndarray,
) -> Dict[str, float]:
    """Compute ADE/FDE shift between baseline and perturbed prediction."""
    disp = np.linalg.norm(pred_baseline - pred_perturbed, axis=-1)
    return {
        "ade_shift": float(np.mean(disp)),
        "fde_shift": float(disp[-1]),
    }
