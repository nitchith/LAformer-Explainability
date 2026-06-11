# LAformer Explainability Analysis

Post-hoc explainability analysis for [LAformer](https://github.com/mengmengliu1998/LAformer) (Lane-Aware Transformer), a trajectory prediction model for autonomous driving.

This project applies perturbation-based explainability methods to understand **which input features drive LAformer's predictions** on the [nuScenes](https://www.nuscenes.org/) dataset.

## Explainability Methods

### Neighbor-Agent Ablation (Leave-One-Out)
For each neighbor agent, a perturbed copy of the input is created with that agent's polyline removed from the vectorized scene representation. The model re-predicts on the perturbed input, and the ADE/FDE shift from baseline quantifies that agent's influence.

### Group Ablation
Same perturbation approach applied to entire groups:
- **No Pedestrians** -- all pedestrian polylines removed
- **No Vehicles** -- all vehicle polylines removed
- **No Agents** -- all neighbor polylines removed (ego trajectory only)
- **No Lanes** -- all lane polyline features zeroed out (structure preserved for encoder compatibility)
- **Ego Only** -- no agents + no lanes (only ego history remains)

### Temporal Occlusion
Each ego history timestep is individually masked (features zeroed) and the prediction shift is measured, revealing which moments in the ego's past are most critical for prediction.

## Project Structure

```
LAformer-Explainability/
├── README.md
├── environment.yml               # Conda environment
├── app.py                        # Interactive Gradio demo (localhost:7860)
├── slides.md                     # Marp presentation slides
├── .gitignore
│
├── checkpoints/
│   └── nuScene_k5/
│       └── model.50.bin          # Pretrained LAformer (2MB)
│
├── data/
│   ├── mini_val_cache/
│   │   └── eval.ex_list          # 61 preprocessed val samples
│   └── mini_train_cache/
│       └── ex_list               # 742 preprocessed train samples
│
├── laformer/                     # LAformer model source (from upstream)
│   ├── model_main.py
│   ├── dataloader_nuscenes.py
│   ├── modeling/
│   │   ├── vectornet.py
│   │   ├── global_graph.py
│   │   ├── laplace_decoder.py
│   │   └── motion_refinement.py
│   └── utils_files/
│       ├── utils.py
│       ├── config.py
│       ├── loss.py
│       └── eval_metrics.py
│
├── explainability/
│   ├── explain_utils.py          # Core utilities (model load, ablation, metrics)
│   ├── run_explainability.py     # Full analysis runner (CLI)
│   └── preview_all.py            # Batch preview generator (CLI)
│
├── preprocessing/
│   └── preprocess_nuscenes.py    # Preprocess raw nuScenes into LAformer format
│
└── results/                      # Generated output
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd LAformer-Explainability
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate laformer-explain
```

### 3. Download nuScenes mini dataset

Download from https://www.nuscenes.org/nuscenes#download (requires free account):
- **v1.0-mini** (~4GB) -- metadata + sensor data
- **nuScenes-map-expansion-v1.3** (~200MB) -- HD map data

Extract both into a `nuscenes_mini/` directory at the project root:

```
nuscenes_mini/
├── maps/
│   └── expansion/
├── samples/
├── sweeps/
└── v1.0-mini/
```

> **Note:** The preprocessed data caches are already included in `data/`, so you only need nuScenes if you want to run the explainability analysis (which uses GT annotations for agent type classification) or regenerate the caches.

### 4. Verify the setup

```bash
# Quick test -- generate val set previews
python -m explainability.preview_all
```

## Usage

All commands should be run from the project root with the conda environment activated.

### Interactive Gradio App

The interactive demo lets you explore explainability results in real time through a browser UI.

```bash
python app.py
```

Open http://localhost:7860 in your browser. The app loads the model and data on startup (~5 seconds).

**Workflow:**

1. **Select a split** (`mini_val` or `mini_train`) and a **sample** from the dropdown (shows scene name, agent count, and FDE).
2. The **baseline prediction** appears on the left with labeled agents (`#0 V`, `#1 P`, ...) and lane indices (`L0`, `L1`, ...).
3. Use the **ablation controls** on the right to run "what-if" experiments:
   - **Group toggles** -- check "Remove Pedestrians", "Remove Vehicles", or "Remove Lanes" to ablate entire categories.
   - **Individual agents** -- enter agent indices (e.g. `0,3,5`) to remove specific neighbors.
   - **Selective lanes** -- enter lane indices (e.g. `0-5,10,12`) to remove specific lane polylines.
4. Click **Run Ablation** to see the ablated prediction side-by-side with the baseline, plus ADE/FDE shift metrics.
5. Use the **Zoom slider** (-2 = close-up, +2 = full scene) to adjust the view without re-running inference.
6. Toggle **Show lane labels** on/off for clarity.

### Generate Preview Images

Preview images show the baseline prediction (all 5 modes) along with ground truth and all neighbor agents, giving a quick visual overview of each sample.

```bash
# Validation set (61 samples, 2 scenes):
python -m explainability.preview_all

# Training set (742 samples, 8 scenes):
python -m explainability.preview_all \
    --cache_path data/mini_train_cache/ex_list \
    --output_dir results/previews/train
```

Output: one PNG per sample + `preview_summary.json` with ADE/FDE metrics.

### Run Explainability Analysis

Full analysis on a single sample: neighbor ablation + group ablation + temporal occlusion.

```bash
# By sample index (val set):
python -m explainability.run_explainability --sample_idx 23

# By scene name (picks sample with most agents):
python -m explainability.run_explainability --scene scene-0103

# On a training set sample:
python -m explainability.run_explainability \
    --sample_idx 545 \
    --cache_path data/mini_train_cache/ex_list

# Skip temporal occlusion (faster):
python -m explainability.run_explainability \
    --sample_idx 23 --skip_temporal
```

Output (saved to `results/explainability/<scene>_idx<N>/`):

| File | Description |
|------|-------------|
| `six_panel_ablation.png` | 2x3 grid: Baseline, No Peds, No Vehs, No Agents, No Lanes, Ego Only |
| `top3_and_ranking.png` | Top-3 most influential neighbors + ranking bar chart |
| `group_ablation_summary.png` | Group ablation shift comparison + influence vs. distance scatter |
| `temporal_occlusion.png` | Perturbed predictions per masked timestep + importance bar chart |
| `summary.json` | All numerical results in machine-readable format |

### Preprocessing (Optional)

The preprocessed data caches are already included. To regenerate from raw nuScenes:

```bash
# Validation split:
python -m preprocessing.preprocess_nuscenes --split mini_val

# Training split:
python -m preprocessing.preprocess_nuscenes --split mini_train
```

## CLI Reference

### `explainability.preview_all`

| Flag | Default | Description |
|------|---------|-------------|
| `--cache_path` | `data/mini_val_cache/eval.ex_list` | Preprocessed sample cache |
| `--output_dir` | `results/previews/val` | Output directory |
| `--checkpoint` | `checkpoints/nuScene_k5/model.50.bin` | Model checkpoint |
| `--nusc_datadir` | `nuscenes_mini` | nuScenes dataset root |
| `--device` | `cpu` | `cpu` or `mps` (Apple Silicon) |

### `explainability.run_explainability`

| Flag | Default | Description |
|------|---------|-------------|
| `--sample_idx` | -- | Sample index in the cache (0-based) |
| `--scene` | -- | nuScenes scene name (e.g. `scene-0103`) |
| `--n_agents` | -- | Filter by exact agent count |
| `--cache_path` | `data/mini_val_cache/eval.ex_list` | Sample cache |
| `--checkpoint` | `checkpoints/nuScene_k5/model.50.bin` | Model checkpoint |
| `--nusc_datadir` | `nuscenes_mini` | nuScenes dataset root |
| `--output_dir` | `results/explainability` | Output directory |
| `--device` | `cpu` | `cpu` or `mps` |
| `--skip_temporal` | `false` | Skip temporal occlusion |

### `preprocessing.preprocess_nuscenes`

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `mini_val` | `mini_train` or `mini_val` |
| `--dataroot` | `nuscenes_mini` | nuScenes dataset root |
| `--version` | `v1.0-mini` | nuScenes version |
| `--output_dir` | auto | Output cache directory |

## Data Summary

### Validation set (61 samples)
| Scene | Samples | Agents | Description |
|-------|---------|--------|-------------|
| scene-0103 | 43 | 21-51 | Pedestrians, turning car, bike rack, cyclist |
| scene-0916 | 18 | 40-59 | Parking lot, bus, many pedestrians |

### Training set (742 samples)
| Scene | Samples | Agents | Description |
|-------|---------|--------|-------------|
| scene-0061 | 45 | 19-50 | Construction, intersection, turn left |
| scene-0553 | 145 | 1-28 | Intersection, peds crossing crosswalk |
| scene-0655 | 23 | 39-53 | Parking lot, jaywalker, bendy bus |
| scene-0757 | 51 | 1-12 | Busy intersection, bus, bicycle |
| scene-0796 | 147 | 1-19 | Scooter, bus, truck, bicycle |
| scene-1077 | 128 | 1-14 | Night, bus stop, high speed |
| scene-1094 | 90 | 8-63 | Night, many peds, jaywalker |
| scene-1100 | 113 | 0-23 | Night, peds, scooter |

## Acknowledgments

The LAformer model code in `laformer/` is from [LAformer: Lane-Aware Transformer for Trajectory Prediction](https://github.com/mengmengliu1998/LAformer) by Mengmeng Liu et al. (CVPR 2023), licensed under the Apache License 2.0.

```
@inproceedings{liu2024laformer,
  title={LAformer: Trajectory Prediction for Autonomous Driving with Lane-Aware Scene Constraints},
  author={Liu, Mengmeng and Cheng, Hao and Chen, Lin and Brber, Hellward and Liniger, Alexander and Van Gool, Luc},
  booktitle={CVPR},
  year={2024}
}
```

The nuScenes dataset is provided by [Motional](https://www.nuscenes.org/) under CC BY-NC-SA 4.0.
