"""
inference.py — incarca modelele o singura data la startup,
expune run_multiclass() si run_pipeline() pentru views.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

try:
    from ultralytics import YOLO as _YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False

# ── constante ────────────────────────────────────────────────────────────────
# IMPORTANT: ordinea cu care AU FOST ANTRENATE modelele (id-ul prezis -> clasa).
# Sursa: data.yaml `names: ["comedo 0", "nodul 3", "papule 1", "pustule 2"]`
# si checkpoint EfficientNet `classes: ['comedo','nodul','papule','pustule']`.
# Deci: 0=comedo, 1=nodul, 2=papula(papule), 3=pustula(pustule).
MODEL_CLASSES = ['comedo', 'nodul', 'papula', 'pustula']

# Ordinea de AFISARE (severitate crescatoare) — folosita pt. IGA/grafice/tabele.
CLASSES     = ['comedo', 'papula', 'pustula', 'nodul']
IGA_WEIGHTS = {'comedo': 1, 'papula': 2, 'pustula': 3, 'nodul': 4}

COLORS_BGR = {
    'comedo':  (  0, 180, 255),
    'papula':  ( 50, 200,  50),
    'pustula': ( 80,  80, 255),
    'nodul':   (255,  50, 150),
}

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models_dir'

_models = {}


def _get_yolo_multi():
    if 'yolo_multi' not in _models:
        path = MODELS_DIR / 'yolo_multiclass.pt'
        if not ULTRALYTICS_OK:
            raise RuntimeError("ultralytics nu este instalat. Ruleaza: pip install ultralytics")
        if not path.exists():
            raise FileNotFoundError(f"Model negasit: {path}")
        _models['yolo_multi'] = _YOLO(str(path))
    return _models['yolo_multi']


def _get_yolo_single():
    if 'yolo_single' not in _models:
        path = MODELS_DIR / 'yolo_singleclass.pt'
        if not ULTRALYTICS_OK:
            raise RuntimeError("ultralytics nu este instalat.")
        if not path.exists():
            raise FileNotFoundError(f"Model negasit: {path}")
        _models['yolo_single'] = _YOLO(str(path))
    return _models['yolo_single']


def _get_efficientnet():
    if 'efficientnet' not in _models:
        # accepta atat .pt cat si .pth
        path = None
        for ext in ('efficientnet_b0.pth', 'efficientnet_b0.pt'):
            candidate = MODELS_DIR / ext
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"Model EfficientNet negasit in {MODELS_DIR}. "
                "Pune fisierul ca efficientnet_b0.pth sau efficientnet_b0.pt"
            )
        if not TIMM_OK:
            raise RuntimeError(
                "timm nu este instalat. Ruleaza: pip install timm"
            )

        # Modelul e antrenat cu timm — folosim timm.create_model
        # num_classes=0 creaza backbone fara capul de clasificare original
        model = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=len(CLASSES)
        )

        checkpoint = torch.load(str(path), map_location='cpu')

        # Checkpoint poate fi fie state_dict direct, fie un dict cu chei extra
        # (epoch, model_state, val_acc, val_loss, classes, arch etc.)
        if isinstance(checkpoint, dict):
            for key in ('model_state', 'state_dict', 'model_state_dict', 'model'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _models['efficientnet'] = model
    return _models['efficientnet']


# Crop EfficientNet — parametri identici cu scriptul de experiment
# (pipeline_two_stage+yolo.py): 20% context, pătrat 224, BICUBIC.
CONTEXT   = 0.20
CROP_SIZE = 224

# Transform fără resize — redimensionarea o face PIL (BICUBIC) in _make_square_crop.
_effnet_tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── utilitare ────────────────────────────────────────────────────────────────
def _make_square_crop(img_np: np.ndarray, x1, y1, x2, y2,
                      context: float = CONTEXT, size: int = CROP_SIZE):
    """
    Identic cu make_square_crop din scriptul de experiment:
      1. extinde cutia cu `context` (20%) pe fiecare latura
      2. construieste un PATRAT centrat pe centrul cutiei (latura = max(w,h))
         folosind PIXELI REALI din imagine, cu zero-padding DOAR unde patratul
         iese din imagine
      3. redimensioneaza la `size`x`size` cu BICUBIC
    Returneaza un PIL.Image sau None daca crop-ul e invalid.
    """
    img_h, img_w = img_np.shape[:2]
    bw = x2 - x1; bh = y2 - y1
    pad_x = bw * context; pad_y = bh * context
    x1e = x1 - pad_x; y1e = y1 - pad_y
    x2e = x2 + pad_x; y2e = y2 + pad_y
    cw = x2e - x1e; ch = y2e - y1e
    side = max(cw, ch)
    cx = (x1e + x2e) / 2; cy = (y1e + y2e) / 2
    x1s = cx - side / 2; y1s = cy - side / 2
    x2s = cx + side / 2; y2s = cy + side / 2

    pl = max(0.0, -x1s); pt = max(0.0, -y1s)
    pr = max(0.0, x2s - img_w); pb = max(0.0, y2s - img_h)

    x1c = int(max(0, round(x1s))); y1c = int(max(0, round(y1s)))
    x2c = int(min(img_w, round(x2s))); y2c = int(min(img_h, round(y2s)))
    if x2c <= x1c or y2c <= y1c:
        return None

    crop = img_np[y1c:y2c, x1c:x2c].copy()
    pl_i, pt_i, pr_i, pb_i = (int(round(v)) for v in (pl, pt, pr, pb))
    if pl_i > 0 or pt_i > 0 or pr_i > 0 or pb_i > 0:
        crop = np.pad(crop, ((pt_i, pb_i), (pl_i, pr_i), (0, 0)),
                      mode="constant", constant_values=0)

    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def _draw_boxes(img_bgr: np.ndarray, detections: list,
                show_labels: bool = True) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d['bbox']]
        color = COLORS_BGR.get(d['class'], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{d['class']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            #cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, y1), color, -1)
            #cv2.putText(out, label, (x1 + 2, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _dashed_rect(img, p1, p2, color, thickness=2, dash=9, gap=6):
    """Deseneaza un dreptunghi cu linie punctata (OpenCV nu are nativ)."""
    x1, y1 = p1
    x2, y2 = p2

    def _line(a, b):
        (ax, ay), (bx, by) = a, b
        dist = int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
        if dist == 0:
            return
        step = dash + gap
        for i in range(0, dist, step):
            s = i / dist
            e = min(i + dash, dist) / dist
            sx = int(ax + (bx - ax) * s); sy = int(ay + (by - ay) * s)
            ex = int(ax + (bx - ax) * e); ey = int(ay + (by - ay) * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    _line((x1, y1), (x2, y1))
    _line((x2, y1), (x2, y2))
    _line((x2, y2), (x1, y2))
    _line((x1, y2), (x1, y1))


def load_gt_boxes(label_path: str, width: int, height: int) -> list:
    """
    Citeste un fisier label YOLO (cls cx cy w h, normalizate) si returneaza
    cutiile in coordonate pixel + numele clasei (mapat cu MODEL_CLASSES).
    """
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = (cx - bw / 2) * width;  y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width;  y2 = (cy + bh / 2) * height
            cls = MODEL_CLASSES[cid] if 0 <= cid < len(MODEL_CLASSES) else str(cid)
            boxes.append({'class': cls, 'bbox': [round(x1, 1), round(y1, 1),
                                                 round(x2, 1), round(y2, 1)]})
    return boxes


def draw_gt_overlay(img_bgr: np.ndarray, gt_boxes: list) -> np.ndarray:
    """
    Deseneaza cutiile de ground truth PESTE o imagine (de obicei deja adnotata
    cu predictii). GT = linie punctata, colorata pe clasa, ca sa se distinga de
    predictii (linie solida).
    """
    out = img_bgr.copy()
    for b in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in b['bbox']]
        color = COLORS_BGR.get(b['class'], (220, 220, 220))
        _dashed_rect(out, (x1, y1), (x2, y2), color, thickness=2)
    return out


def _iga_score(detections: list) -> dict:
    counts = {c: 0 for c in CLASSES}
    for d in detections:
        counts[d['class']] += 1
    score = sum(counts[c] * IGA_WEIGHTS[c] for c in CLASSES)
    breakdown = [
        {'class': c, 'count': counts[c],
         'weight': IGA_WEIGHTS[c],
         'subtotal': counts[c] * IGA_WEIGHTS[c]}
        for c in CLASSES
    ]
    return {'score': score, 'breakdown': breakdown}


def _severity_label(score: int) -> str:
    if score == 0:   return 'Fara leziuni'
    elif score <= 5:  return 'Usor'
    elif score <= 20: return 'Moderat'
    elif score <= 50: return 'Sever'
    else:             return 'Foarte sever'


# ── inferenta publica ────────────────────────────────────────────────────────
def run_multiclass(image_path: str, conf: float = 0.10,
                   iou: float = 0.40, tta: bool = False) -> dict:
    yolo    = _get_yolo_multi()
    results = yolo(image_path, conf=conf, iou=iou, augment=tta, verbose=False)[0]
    img_bgr = cv2.imread(image_path)

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        detections.append({
            'class':      MODEL_CLASSES[cls_id],
            'confidence': round(float(box.conf), 4),
            'bbox':       [round(v, 1) for v in box.xyxy[0].tolist()],
        })

    annotated = _draw_boxes(img_bgr, detections)
    iga = _iga_score(detections)
    iga['severity'] = _severity_label(iga['score'])

    return {
        'detections': detections,
        'annotated':  annotated,
        'raw':        img_bgr,
        'iga':        iga,
        'n_det':      len(detections),
        'mode':       'YOLO Multiclass (cls_pw=0.5)',
    }


def run_pipeline(image_path: str, conf: float = 0.15,
                 iou: float = 0.45, tta: bool = False) -> dict:
    yolo   = _get_yolo_single()
    effnet = _get_efficientnet()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo(image_path, conf=conf, iou=iou, augment=tta, verbose=False)[0]

    detections = []
    for box in results.boxes:
        # coordonate float pt. crop (ca in script), int doar pt. desen
        fx1, fy1, fx2, fy2 = box.xyxy[0].tolist()
        det_conf = float(box.conf)

        patch_pil = _make_square_crop(img_rgb, fx1, fy1, fx2, fy2)
        if patch_pil is None:
            continue
        tensor = _effnet_tf(patch_pil).unsqueeze(0)

        with torch.no_grad():
            logits   = effnet(tensor)
            probs    = torch.softmax(logits, dim=1)[0]
            cls_id   = int(probs.argmax())
            cls_conf = float(probs[cls_id])

        detections.append({
            'class':      MODEL_CLASSES[cls_id],
            'confidence': round(det_conf * cls_conf, 4),
            'det_conf':   round(det_conf, 4),
            'cls_conf':   round(cls_conf, 4),
            'bbox':       [int(fx1), int(fy1), int(fx2), int(fy2)],
        })

    annotated = _draw_boxes(img_bgr, detections)
    iga = _iga_score(detections)
    iga['severity'] = _severity_label(iga['score'])

    return {
        'detections': detections,
        'annotated':  annotated,
        'raw':        img_bgr,
        'iga':        iga,
        'n_det':      len(detections),
        'mode':       'Pipeline Hibrid (YOLO + EfficientNet-B0)',
    }
