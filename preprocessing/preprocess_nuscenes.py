"""
Preprocess nuScenes mini data into LAformer's vectorized format.

Runs single-threaded to avoid macOS multiprocessing pickle issues.

Usage:
    cd <project_root>
    conda activate laformer-explain

    # Validation split (61 samples):
    python -m preprocessing.preprocess_nuscenes --split mini_val

    # Training split (742 samples):
    python -m preprocessing.preprocess_nuscenes --split mini_train
"""

import argparse
import os
import pickle
import sys
import zlib
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "laformer"))

from nuscenes import NuScenes
from dataloader_nuscenes import NuScenesData


def main():
    parser = argparse.ArgumentParser(description="Preprocess nuScenes for LAformer")
    parser.add_argument("--split", type=str, default="mini_val",
                        choices=["mini_train", "mini_val"],
                        help="nuScenes prediction split")
    parser.add_argument("--dataroot", type=str,
                        default=str(PROJ_ROOT / "nuscenes_mini"),
                        help="Path to nuScenes dataset root")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/mini_{train,val}_cache)")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "train" if "train" in args.split else "val"
        args.output_dir = str(PROJ_ROOT / "data" / f"mini_{suffix}_cache")

    ds_args = {
        "t_h": 2,
        "t_f": 6,
        "split": args.split,
        "dataroot": args.dataroot,
        "cores": 1,
        "vis_mode": "rel",
        "show_fig": False,
        "fig_dir": "./debug",
        "vis_func": "visualize",
        "img_only": False,
        "debug": False,
        "mapping_only": True,
    }

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nuscenes = NuScenes(args.version, dataroot=args.dataroot)

    dataset = NuScenesData(
        mode="extract_data", data_dir=args.output_dir,
        args=ds_args, nuscenes=nuscenes,
    )
    print(f"Total tokens to process: {len(dataset)}")

    ex_list = []
    skipped = 0
    for idx in range(len(dataset)):
        mapping = dataset[idx]
        if mapping is None or len(mapping) == 0:
            skipped += 1
            continue
        data_compress = zlib.compress(pickle.dumps(mapping))
        ex_list.append(data_compress)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1} / {len(dataset)} processed "
                  f"({len(ex_list)} valid, {skipped} skipped)")

    print(f"\nDone: {len(ex_list)} valid samples out of {len(dataset)} tokens "
          f"({skipped} skipped)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = "eval.ex_list" if "val" in args.split else "ex_list"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(ex_list, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
