"""
views.py — gestionează upload-ul imaginii, apelează inferența,
returnează rezultatele ca JSON (pentru fetch din JS) sau HTML.
"""

import base64
import os
import uuid

import cv2
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from django.core.files.uploadedfile import InMemoryUploadedFile

from .inference import (
    run_multiclass, run_pipeline,
    load_gt_boxes, draw_gt_overlay, CLASSES,
)

# Folder uploads compatibil Windows si Linux
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'uploads_tmp'
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_SIZE_MB = 20


def _validate_and_save(request):
    """
    Valideaza fisierul imagine din request si il salveaza temporar.
    Returneaza tuple (tmp_path, ext, None) la succes
    sau (None, None, JsonResponse_eroare) la esec.
    Apelantul e responsabil sa stearga tmp_path la final.
    """
    if 'image' not in request.FILES:
        return None, None, JsonResponse({'error': 'Nicio imagine trimisa.'}, status=400)

    uploaded: InMemoryUploadedFile = request.FILES['image']
    ext = os.path.splitext(uploaded.name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return None, None, JsonResponse(
            {'error': f'Format nesuportat: {ext}. Acceptat: JPG, PNG, WEBP, BMP.'},
            status=400
        )

    if uploaded.size > MAX_SIZE_MB * 1024 * 1024:
        return None, None, JsonResponse(
            {'error': f'Imaginea depaseste {MAX_SIZE_MB}MB.'},
            status=400
        )

    tmp_name = f"{uuid.uuid4()}{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    with open(tmp_path, 'wb') as f:
        for chunk in uploaded.chunks():
            f.write(chunk)

    return tmp_path, ext, None


def _b64(img_bgr):
    """Encodeaza o imagine BGR ca JPEG base64."""
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')


def _encode_result(result):
    """Serializeaza un rezultat de inferenta in dict JSON-ready (cu imagini base64)."""
    return {
        'mode':          result['mode'],
        'n_det':         result['n_det'],
        'detections':    result['detections'],
        'iga':           result['iga'],
        'image_b64':     _b64(result['annotated']),  # cu predictii
        'image_raw_b64': _b64(result['raw']),        # imaginea bruta
    }


def _find_gt_label(filename: str):
    """
    Cauta un fisier label YOLO (.txt) cu acelasi nume (fara extensie) ca imaginea
    incarcata, in folderele configurate (settings.GT_LABELS_DIRS).
    Returneaza calea sau None.
    """
    if not filename:
        return None
    stem = os.path.splitext(os.path.basename(filename))[0]
    for d in getattr(settings, 'GT_LABELS_DIRS', []):
        candidate = os.path.join(d, stem + '.txt')
        if os.path.exists(candidate):
            return candidate
    return None


def _load_gt(raw_bgr, label_path):
    """Incarca cutiile GT + numara pe clasa. Returneaza (gt_boxes, counts) sau (None, None)."""
    h, w = raw_bgr.shape[:2]
    gt_boxes = load_gt_boxes(label_path, w, h)
    if not gt_boxes:
        return None, None
    counts = {c: 0 for c in CLASSES}
    for b in gt_boxes:
        if b['class'] in counts:
            counts[b['class']] += 1
    return gt_boxes, counts


def _build_gt_payload(raw_bgr, annotated_bgr, label_path):
    """
    Construieste campurile GT pentru modul single. Returneaza:
      has_gt, n_gt, gt_counts,
      image_gt_only_b64  (GT punctat peste imaginea bruta),
      image_pred_gt_b64  (predictii + GT).
    """
    gt_boxes, counts = _load_gt(raw_bgr, label_path)
    if gt_boxes is None:
        return {'has_gt': False}
    return {
        'has_gt':            True,
        'n_gt':              len(gt_boxes),
        'gt_counts':         counts,
        'image_gt_only_b64': _b64(draw_gt_overlay(raw_bgr, gt_boxes)),
        'image_pred_gt_b64': _b64(draw_gt_overlay(annotated_bgr, gt_boxes)),
    }


@ensure_csrf_cookie
def index(request):
    """Serveste pagina principala si seteaza cookie-ul CSRF."""
    return render(request, 'detector/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def analyze(request):
    """
    Endpoint principal de analiza.
    Primeste: multipart/form-data cu campurile:
      - image: fisierul imagine
      - mode: 'multiclass' | 'pipeline'
      - conf: float
      - iou:  float
    Returneaza: JSON cu detectiile, imaginea adnotata (base64), scorul IGA.
    """
    # numele original (pt. cautarea ground truth dupa nume)
    orig_name = request.FILES['image'].name if 'image' in request.FILES else ''

    # validare + salvare fisier
    tmp_path, ext, err = _validate_and_save(request)
    if err is not None:
        return err

    # parsare parametri
    mode = request.POST.get('mode', 'multiclass')
    if mode not in ('multiclass', 'pipeline'):
        mode = 'multiclass'

    try:
        conf = float(request.POST.get('conf', 0.10 if mode == 'multiclass' else 0.15))
        iou  = float(request.POST.get('iou',  0.40 if mode == 'multiclass' else 0.45))
        conf = max(0.01, min(0.99, conf))
        iou  = max(0.10, min(0.99, iou))
    except (ValueError, TypeError):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return JsonResponse({'error': 'Parametri conf/iou invalizi.'}, status=400)

    tta = request.POST.get('tta', '').lower() in ('1', 'true', 'on', 'yes')

    try:
        # rulare inferenta
        try:
            if mode == 'multiclass':
                result = run_multiclass(tmp_path, conf=conf, iou=iou, tta=tta)
            else:
                result = run_pipeline(tmp_path, conf=conf, iou=iou, tta=tta)
        except FileNotFoundError as e:
            return JsonResponse(
                {'error': f'Model negasit: {e}. Verifica folderul models_dir/.'},
                status=500
            )
        except RuntimeError as e:
            return JsonResponse({'error': str(e)}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Eroare la inferenta: {str(e)}'}, status=500)

        payload = {
            'success': True,
            **_encode_result(result),
            'params': {'conf': conf, 'iou': iou, 'mode': mode, 'tta': tta},
        }

        # ground truth (overlay) daca exista un label cu acelasi nume
        gt_path = _find_gt_label(orig_name)
        if gt_path:
            try:
                payload.update(_build_gt_payload(result['raw'], result['annotated'], gt_path))
            except Exception:
                payload['has_gt'] = False  # GT optional, nu blocheaza analiza
        else:
            payload['has_gt'] = False

        return JsonResponse(payload)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@csrf_exempt
@require_http_methods(["POST"])
def compare(request):
    """
    Endpoint de comparatie: ruleaza AMBELE modele pe aceeasi imagine
    si returneaza rezultatele side-by-side.
    Fiecare model foloseste pragurile lui implicite (optime) ca sa fie
    o comparatie corecta, dar pot fi suprascrise prin parametri.
    Returneaza: JSON cu { multiclass: {...}, pipeline: {...} }.
    """
    orig_name = request.FILES['image'].name if 'image' in request.FILES else ''

    tmp_path, ext, err = _validate_and_save(request)
    if err is not None:
        return err

    # praguri: implicite per model, suprascriibile optional
    def _f(name, default):
        try:
            return max(0.01, min(0.99, float(request.POST.get(name, default))))
        except (ValueError, TypeError):
            return default

    conf_mc = _f('conf_mc', 0.10)
    iou_mc  = _f('iou_mc',  0.40)
    conf_pl = _f('conf_pl', 0.15)
    iou_pl  = _f('iou_pl',  0.45)
    tta = request.POST.get('tta', '').lower() in ('1', 'true', 'on', 'yes')

    try:
        try:
            res_mc = run_multiclass(tmp_path, conf=conf_mc, iou=iou_mc, tta=tta)
            res_pl = run_pipeline(tmp_path, conf=conf_pl, iou=iou_pl, tta=tta)
        except FileNotFoundError as e:
            return JsonResponse(
                {'error': f'Model negasit: {e}. Verifica folderul models_dir/.'},
                status=500
            )
        except RuntimeError as e:
            return JsonResponse({'error': str(e)}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Eroare la inferenta: {str(e)}'}, status=500)

        out = {
            'success':    True,
            'multiclass': _encode_result(res_mc),
            'pipeline':   _encode_result(res_pl),
            'has_gt':     False,
            'params': {
                'conf_mc': conf_mc, 'iou_mc': iou_mc,
                'conf_pl': conf_pl, 'iou_pl': iou_pl, 'tta': tta,
            },
        }

        # ground truth comun (acelasi pt. ambele modele) + overlay per model
        gt_path = _find_gt_label(orig_name)
        if gt_path:
            try:
                gt_boxes, counts = _load_gt(res_mc['raw'], gt_path)
                if gt_boxes is not None:
                    out['has_gt']    = True
                    out['n_gt']      = len(gt_boxes)
                    out['gt_counts'] = counts
                    out['image_gt_only_b64'] = _b64(draw_gt_overlay(res_mc['raw'], gt_boxes))
                    out['multiclass']['image_pred_gt_b64'] = _b64(
                        draw_gt_overlay(res_mc['annotated'], gt_boxes))
                    out['pipeline']['image_pred_gt_b64'] = _b64(
                        draw_gt_overlay(res_pl['annotated'], gt_boxes))
            except Exception:
                out['has_gt'] = False

        return JsonResponse(out)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
