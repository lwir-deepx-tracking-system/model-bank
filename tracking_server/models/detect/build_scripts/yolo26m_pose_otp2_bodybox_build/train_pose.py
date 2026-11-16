"""Train a YOLO26 pose model with the "keep out-of-picture keypoints" changes of pose_partial.py (close_range_20261113/PLAN.md, experiment A onward).

Usage (GPU machine):
  python train_pose.py --data <data.yaml> --weights <yolo26m-pose.pt> --project <runs dir outside OneDrive> --name <unique run name>
                       [--epochs 60] [--batch 32] [--imgsz 640] [--device 0] [--partial-p 0.5] [--fraction 1.0] [--workers 8] [--wandb on|off]

Smoke test first (rule of training/flir_yolo26/AGENTS.md): --epochs 1 --fraction 0.05 --name <name>_smoke, then look at the saved files.
--wandb on stops if W&B is not logged in on this machine (no silent offline fallback). --wandb off writes only the local results.csv.
The run folder gets RUN_MANIFEST.json (arguments, library versions, sha256 of the weights, this file, pose_partial.py and the data yaml).
Augmentation: mosaic off (see pose_partial.py), affine scale 0.5 / translate 0.2 / rotate 5 deg, left-right flip, brightness only (thermal = grey).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_partial  # noqa: E402


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--weights', required=True)
    ap.add_argument('--project', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--batch', type=int, default=32); ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0'); ap.add_argument('--partial-p', type=float, default=0.5); ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--wandb', choices=('on', 'off'), required=True)
    # "fast" (user, 11-13): mixed precision, images cached in RAM, cuDNN auto-tuning instead of bit-for-bit repeatability, early stop when val stalls
    ap.add_argument('--fast', type=int, default=1); ap.add_argument('--patience', type=int, default=15)
    # body-box mode (user, 11-13): --template <template_points.npy> makes the box label the unclipped body box (pose_partial.BODY)
    ap.add_argument('--template', default=None)
    ap.add_argument('--occlude-p', type=float, default=0.0)        # experiment C: objects in front of people (pose_partial.Occlude)
    ap.add_argument('--thermal-look', type=int, default=0)         # blur / noise / gain changes (pose_partial.ThermalLook)
    # Fine-tuning settings. The first run (optimizer=auto -> AdamW 0.002, warmup_bias_lr 0.1 for 3 epochs) fell from pose mAP 0.79 to 0.20 after
    # epoch 1 and needed 12 epochs to get back to 0.65. Lower rate, short warm-up, no bias boost.
    ap.add_argument('--lr0', type=float, default=0.0005); ap.add_argument('--warmup-epochs', type=float, default=1.0)
    a = ap.parse_args()
    import torch
    import ultralytics
    from ultralytics import settings
    if a.wandb == 'on':
        import wandb
        if wandb.api.api_key is None: raise SystemExit('W&B is not logged in on this machine: run `wandb login` first (no offline fallback).')
    settings.update(wandb=a.wandb == 'on')
    run = Path(a.project) / a.name
    if run.exists(): raise SystemExit('run folder already exists: %s (use a new --name)' % run)
    Trainer = pose_partial.make_trainer(partial_p=a.partial_p, body_box_template=a.template, occlude_p=a.occlude_p, thermal_look=bool(a.thermal_look))
    over = dict(model=a.weights, data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device=a.device, workers=a.workers, seed=a.seed,
                project=a.project, name=a.name, exist_ok=False, fraction=a.fraction, plots=True, save_period=10, patience=a.patience,
                amp=True, deterministic=not a.fast, cache='ram' if a.fast else False,
                optimizer='AdamW', lr0=a.lr0, lrf=0.1, warmup_epochs=a.warmup_epochs, warmup_bias_lr=0.0,
                mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, scale=0.5, translate=0.2, degrees=5.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0,
                hsv_v=0.0 if a.thermal_look else 0.4)          # ThermalLook does the brightness changes itself
    t = Trainer(overrides=over)
    manifest = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), args=vars(a), overrides=over, ultralytics=ultralytics.__version__, torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    sha256=dict(weights=sha(a.weights), train_pose=sha(__file__), pose_partial=sha(Path(__file__).with_name('pose_partial.py')), data_yaml=sha(a.data)),
                    patches=dict(hidden_flag=pose_partial.HIDDEN, area_thr=pose_partial.AREA_THR, occlude_p=a.occlude_p, body_box=a.template is not None, thermal_look=bool(a.thermal_look)))
    t.save_dir.mkdir(parents=True, exist_ok=True)
    (t.save_dir / 'RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    t.train()
    print('TRAIN DONE', t.save_dir, flush=True)


if __name__ == '__main__':
    main()
