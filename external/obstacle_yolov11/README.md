# obstacle_yolov11

YOLOv11 training utilities and scripts. Intended to live as an independent Git repository inside your `catkin_ws` without interfering with ROS builds.

## Place As Subfolder (not in src)
- Recommended location: `catkin_ws/external/obstacle_yolov11` (create `external/` if needed).
- Avoid putting this inside `catkin_ws/src` unless you make it a proper Catkin package (not needed here).

## Keep It Independent From catkin_ws
- In your parent workspace repo (if `catkin_ws` itself is a Git repo), add this path to its `.gitignore` so it won’t be tracked there:
  - If placed at `catkin_ws/external/obstacle_yolov11`, add: `external/obstacle_yolov11/`
  - If placed at `catkin_ws/obstacle_yolov11`, add: `obstacle_yolov11/`

## Initialize As Its Own Repo
```bash
cd obstacle_yolov11
git init
git add .
git commit -m "init: obstacle_yolov11"
# Create an empty GitHub repo first, then:
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Environment
- Python 3.9+ recommended
- Install dependencies (Torch should match your GPU/CUDA setup):
```bash
# Install torch per https://pytorch.org/get-started/locally/
# Then common packages
pip install ultralytics opencv-python pyyaml
```

## Training
Example using the provided script:
```bash
python train_yolov11.py \
  --dataset-root <path_to_dataset_root> \
  --model yolo11n.pt \
  --epochs 100 \
  --img-size 640 \
  --batch 16 \
  --device 0
```
Notes:
- `--dataset-root` must contain: `images/{train,val,test}` and `labels/{train,val,test}`.
- The script writes a dataset YAML into that folder automatically and saves runs to `runs/train/<name>`.
- Checkpoints (including `best.pt`) are saved under `runs/` (ignored by Git here).

## What This Repo Tracks
- Source code and configs needed to train.
- Excludes large data and outputs via `.gitignore`:
  - `runs/`, datasets (`dataset*/`, `raw/`, `augmented*/`), weights (`*.pt`, `*.onnx`, ...), caches.
- If you need to share weights, use a release, separate storage, or Git LFS.

## Moving Under catkin_ws
If this folder isn’t already under `catkin_ws`, move it:
```bash
# Example layout
<path>/catkin_ws/external/obstacle_yolov11
```
Then ensure the parent `.gitignore` is updated as described above.
