# -*- coding: utf-8 -*-
"""Ultralytics replacement for the MATLAB experiment matrix.

Same data, same split, same 300-epoch schedule, same augmentation defaults,
but ~5x faster per epoch and using the official implementation.  Each condition
is trained from scratch (model=yolov8n.yaml), then evaluated on the SAME
held-out test split.  Finished conditions are skipped, so the driver can be
relaunched after an interruption.
"""
import os, sys, json, time, traceback
os.environ.setdefault("YOLO_CONFIG_DIR",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics"))
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "ultralytics_data")
PROJ = os.path.join(BASE, "..", "ultralytics_runs")
SUMMARY = os.path.join(BASE, "yolo_matrix_summary.json")
LOGFILE = os.path.join(BASE, "yolo_matrix.log")

def log(msg):
    """Print and append to a log file, so a scheduled task is observable."""
    with open(LOGFILE, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)

EPOCHS = int(os.environ.get("YOLO_EPOCHS", "300"))
SEL = os.environ.get("YOLO_ONLY", "")      # optional comma-separated filter

#            name             variant   imgsz seed
JOBS = [
    ("clean_640",      "clean",  640, 2026),
    ("jit05_640",      "jit05",  640, 2026),
    ("jit10_640",      "jit10",  640, 2026),
    ("jit20_640",      "jit20",  640, 2026),
    ("drop20_640",     "drop20", 640, 2026),
    ("drop40_640",     "drop40", 640, 2026),
    ("drop60_640",     "drop60", 640, 2026),
    ("clean_640_s27",  "clean",  640, 2027),
    ("clean_640_s28",  "clean",  640, 2028),
    ("drop60_640_s27", "drop60", 640, 2027),
    ("drop60_640_s28", "drop60", 640, 2028),
    ("jit20_640_s27",  "jit20",  640, 2027),
    ("jit20_640_s28",  "jit20",  640, 2028),
    ("clean_320",      "clean",  320, 2026),
    ("jit10_320",      "jit10",  320, 2026),
    ("drop40_320",     "drop40", 320, 2026),
    ("clean_320_s27",  "clean",  320, 2027),
    ("clean_320_s28",  "clean",  320, 2028),
]

summary = {}
if os.path.isfile(SUMMARY):
    with open(SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)

log("==== yolo matrix run %s (epochs=%d) ====" % (time.strftime("%Y-%m-%d %H:%M:%S"), EPOCHS))

only = set(s for s in SEL.split(",") if s) if SEL else None

for name, variant, imgsz, seed in JOBS:
    if only and name not in only:
        continue
    if name in summary:
        log("== %s already finished, skip" % name)
        continue
    run_dir = os.path.join(PROJ, name)
    last_ckpt = os.path.join(run_dir, "weights", "last.pt")
    try:
        t0 = time.time()
        if os.path.isfile(last_ckpt):
            # An interrupted run: continue it rather than starting over.
            log("== %s resuming from last.pt" % name)
            model = YOLO(last_ckpt)
            model.train(resume=True)
        else:
            model = YOLO("yolov8n.yaml")
            model.train(
                data=os.path.join(DATA, variant, "data.yaml"),
                epochs=EPOCHS, imgsz=imgsz, batch=16, workers=0, device=0,
                seed=seed, deterministic=True,
                cos_lr=True,             # match the MATLAB cosine schedule
                amp=False,               # AMP check downloads 5.3 MB and stalls here
                project=PROJ, name=name, exist_ok=True,
                plots=False, val=True, verbose=False,
            )
        m = model.val(data=os.path.join(DATA, variant, "data.yaml"),
                      split="test", imgsz=imgsz, batch=16, workers=0,
                      device=0, plots=False, verbose=False)
        rec = {
            "name": name, "variant": variant, "imgsz": imgsz, "seed": seed,
            "mAP50": float(m.box.map50), "mAP5095": float(m.box.map),
            "perClassAP50": [float(x) for x in m.box.ap50],
            "seconds": time.time() - t0, "epochs": EPOCHS,
        }
        summary[name] = rec
        with open(SUMMARY, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        log("== %s DONE mAP50=%.4f mAP5095=%.4f (%.0f s)" %
            (name, rec["mAP50"], rec["mAP5095"], rec["seconds"]))
    except Exception as exc:                     # keep going, record the failure
        log("== %s FAILED: %s" % (name, exc))
        traceback.print_exc()

log("YOLO MATRIX FINISHED")
