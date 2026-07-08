# LayoutVLM

<div align="left">
    <a href="https://ai.stanford.edu/~sunfanyun/layoutvlm"><img src="https://img.shields.io/badge/🌐 Website-Visit-orange"></a>
    <a href=""><img src="https://img.shields.io/badge/arXiv-PDF-blue"></a>
</div>

<br>

## Installation

1. Clone this repository
2. Install dependencies (python 3.10):
```bash
pip install -r requirements.txt
```
3. Install Rotated IOU Loss (https://github.com/lilanxiao/Rotated\_IoU)
```
cd third_party/Rotated_IoU/cuda_op
python setup.py install
````

## Data preprocessing
1. Download the dataset https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing
2. Unzip it.

Refer to https://github.com/allenai/Holodeck and https://github.com/allenai/objathor for how we preprocess Objaverse assets.

## Usage

1. Prepare a scene configuration JSON file of Objaverse assets with the following structure:
```json
{
    "task_description": ...,
    "layout_criteria": ...,
    "boundary": {
        "floor_vertices": [[x1, y1, z1], [x2, y2, z2], ...],
        "wall_height": height
    },
    "assets": {
        "asset_id": {
            "path": "path/to/asset.glb",
            "assetMetadata": {
                "boundingBox": {
                    "x": width,
                    "y": depth,
                    "z": height
                }
            }
        }
    }
}
```

2. Run LayoutVLM:
```bash
python main.py --scene_json_file path/to/scene.json --openai_api_key your_api_key
```

## Text-to-scene CLI (`cli.py`)

Generate a scene from a textual description instead of a hand-authored scene
JSON. It turns the prompt into a room boundary + asset shopping list (one
LLM call), resolves those against a processed-asset directory, runs the solver,
and writes the result to `outputs/<YYYYMMDD-HHMMSS UTC>/`. The full list of
external local resources this method requires (processed Objaverse assets,
HDRIs, optional asset library, bpy, the Rotated-IoU CUDA op, an LLM endpoint)
and how to prepare each one is documented in the **top docstring of `cli.py`**
(`python cli.py --help`, or read `cli.py`).

```bash
python cli.py \
    --prompt "a cozy beach-inspired bedroom, 4m x 5m, with a queen bed and a rattan chair" \
    --base_url https://api.openai.com/v1 \
    --api_key sk-... \
    --model gpt-4o \
    --temperature 0.0 \
    --asset_dir ./objaverse_processed \
    --hdri_dir ./vlmunr_hdri \
    --variants
```

Each run is a folder `outputs/<YYYYMMDD-HHMMSS UTC>/` containing `config.json`
(prompt + LLM config; the API key is stored **redacted** only) and a `base/`
sub-folder holding the generated scene:

```
outputs/<timestamp>/
  config.json
  base/
    task.json         # resolved + normalized task the solver/renderer consume
    layout.json       # placed instances (the scene)
    scene_spec.json   # the LLM-produced scene spec (boundary + asset list)
    meshes/           # the GLB meshes referenced by the scene, copied in
    renderings/       # only with --render (and bpy); always named "renderings"
    solve_run/        # real mode only: per-group solver artifacts
  variant_01_half/
  variant_02_biggest-only/
  variant_03_scrambled/
  variant_04_worst-object/
```

Every scene folder (`base/` and each variant) has the **same** internal
structure (`task.json`, `layout.json`, `meshes/`, and `renderings/` when
rendered), so each is independently renderable and portable — the referenced
meshes are copied into `meshes/` and the task paths rewritten to `./meshes/…`,
so a folder no longer depends on the external `--asset_dir`.

* `--variants` also writes four content variants as sub-directories, **without
  re-running the LLM or solver** (they fork the base layout):
  - `variant_01_half` — keep round(n/2) instances (seeded)
  - `variant_02_biggest-only` — keep the single largest instance (bbox volume)
  - `variant_03_scrambled` — relocate every instance within the floor polygon,
    preserving z/rotation/identity (objects stay in their region)
  - `variant_04_worst-object` — swap each instance's asset identity to the
    worst-matching candidate from `--asset_library` (requires the library)
* `--mock` skips the LLM + gradient solver and places assets at random floor
  points, so the whole pipeline + variants are exercisable offline without
  GPU/Blender or API cost.
* `--render` renders each scene (base + variants) into its own `renderings/`
  sub-folder via `vlmunr_render`. Requires bpy + HDRIs; degrades gracefully
  (skips rendering, keeps the scene + meshes) if bpy is unavailable or an
  asset mesh is missing. `--render_phase` selects the render phase (default
  `1a` = one image per resolution at the baseline yaw/pitch/hdri).
* `--asset_library path.json` — JSON list of
  `{"category","description","path"[,"assetMetadata":{"boundingBox":...}]}`,
  required for the worst-object variant.

## Output
The script will generate a layout.json file in the specified save directory containing the optimized positions and orientations of all assets in the scene.

## BibTeX
```bibtex
@inproceedings{sun2025layoutvlm,
  title={Layoutvlm: Differentiable optimization of 3d layout via vision-language models},
  author={Sun, Fan-Yun and Liu, Weiyu and Gu, Siyi and Lim, Dylan and Bhat, Goutam and Tombari, Federico and Li, Manling and Haber, Nick and Wu, Jiajun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29469--29478},
  year={2025}
}
```
