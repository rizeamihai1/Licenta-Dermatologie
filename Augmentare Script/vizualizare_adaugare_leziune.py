import cv2
import numpy as np
import os
import random
from pathlib import Path

DATASET_PATH = "."
PATCHES_PATH_NODUL   = "./patches_minoritare_nodul"
PATCHES_PATH_PUSTULE = "./patches_minoritare_pustule"
PATCHES_PATH_COMEDO  = "./patches_minoritare_comedo"

OUTPUT_IMAGES_DIR = "./augmented/train/images"
OUTPUT_LABELS_DIR = "./augmented/train/labels"

AUGMENTARI_PER_IMAGINE = 2

MIN_COMEDO, MAX_COMEDO   = 1, 2
MIN_NODULI,  MAX_NODULI  = 1, 3
MIN_PUSTULE, MAX_PUSTULE = 2, 5

COMEDO_CLASS_ID = 0    # clasa "comedo"
NODUL_CLASS_ID   = 1   # clasa "nodul"
PUSTULA_CLASS_ID = 3   # clasa "pustula"

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp"}
SCALA_MARIRE_MIN   = 0.8
SCALA_MARIRE_MAX   = 1.2

# 1. Script de generare a mastii pielii
def build_skin_mask_dynamic(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    blurred = cv2.GaussianBlur(image_bgr, (11, 11), 0) # o blurez 
    ycrcb   = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb) # o transform din BGR in YCrCb

    cw, ch = w // 2, h // 2
    dw, dh = int(w * 0.1), int(h * 0.1)
    center_patch = ycrcb[ch - dh:ch + dh, cw - dw:cw + dw] # dreptunghiul centrat 20% din imaginea mare => Ma bazez ca subiectul este mereu in centrul imagini

    median_cr = np.median(center_patch[:, :, 1])
    median_cb = np.median(center_patch[:, :, 2])
    margin    = 20

    # iau mediana din Cr si Cb si mai adaug o margine de 20%
    # mediana si nu medie deoarece media este influentata de outlieri

    lower_bound = np.array([0,   max(0,   median_cr - margin), max(0,   median_cb - margin)], dtype=np.uint8)
    upper_bound = np.array([255, min(255, median_cr + margin), min(255, median_cb + margin)], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower_bound, upper_bound)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # Mai treb o data imaginea originala din BGR in Gri pentru a elmina zonele foarte intunecate (ex par inchis, umbre, background)
    mask[gray < 45] = 0

    # operatiile de eroziune si dilarate
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


# ---------------------------------------------------------------------------
# ── 2. LOGICA YOLO ȘI EVITAREA COLIZIUNILOR ────────────────────────────────
# ---------------------------------------------------------------------------
def citeste_etichete_yolo(txt_path, w_img, h_img):
    labels = []
    if not os.path.exists(txt_path):
        return labels
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
                cx, cy = int(cx_norm * w_img), int(cy_norm * h_img)
                bw, bh = int(w_norm * w_img), int(h_norm * h_img)
                x_min, y_min = int(cx - bw / 2), int(cy - bh / 2)
                x_max, y_max = int(cx + bw / 2), int(cy + bh / 2)
                labels.append([class_id, x_min, y_min, x_max, y_max])
    return labels

def creaza_masca_interzisa(labels, w_img, h_img, padding=10):
    forbidden = np.zeros((h_img, w_img), dtype=np.uint8)
    for lbl in labels:
        _, x_min, y_min, x_max, y_max = lbl
        cv2.rectangle(forbidden, (max(0, x_min - padding), max(0, y_min - padding)), 
                      (min(w_img, x_max + padding), min(h_img, y_max + padding)), 255, -1)
    return forbidden

def incarca_set_leziuni(patches_folder):
    patches = []
    folder = Path(patches_folder)
    if not folder.exists():
        print(f"[ATENTIE] Folderul nu exista: {patches_folder}")
        return patches
    for p in folder.iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            img = cv2.imread(str(p))
            if img is not None:
                patches.append(img)
    print(f"  Incarcat {len(patches)} petice din {patches_folder}")
    return patches

# ---------------------------------------------------------------------------
# ── 3. PIPELINE-UL DE VIZUALIZARE ──────────────────────────────────────────
# ---------------------------------------------------------------------------
def vizualizeaza_augmentarea(dataset_path):
    train_img_dir = Path(dataset_path) / "train" / "images"
    train_lbl_dir = Path(dataset_path) / "train" / "labels"

    set_noduli  = incarca_set_leziuni(PATCHES_PATH_NODUL)
    set_pustule = incarca_set_leziuni(PATCHES_PATH_PUSTULE)
    set_comedo  = incarca_set_leziuni(PATCHES_PATH_COMEDO)

    if not set_noduli or not set_pustule or not set_comedo:
        print("Atentie: unul dintre seturile de leziuni este gol.")
        return

    image_paths = sorted([
        p for p in train_img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
    ])

    if not image_paths:
        print("Nu am gasit imagini.")
        return

    current_idx = 0
    total_images = len(image_paths)

    window_name = "Augmentation Debugger (D: inainte | A: inapoi | Q: iesire)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 600)

    while True:
        img_path = image_paths[current_idx]
        img_originala = cv2.imread(str(img_path))

        if img_originala is None:
            current_idx = (current_idx + 1) % total_images
            continue

        h_img, w_img = img_originala.shape[:2]
        txt_path = train_lbl_dir / (img_path.stem + ".txt")

        etichete_existente = citeste_etichete_yolo(txt_path, w_img, h_img)
        masca_piele = build_skin_mask_dynamic(img_originala)
        masca_interzisa = creaza_masca_interzisa(etichete_existente, w_img, h_img)

        zona_sigura = cv2.bitwise_and(masca_piele, cv2.bitwise_not(masca_interzisa))

        # copie a mastii inainte sa taiem din ea => asta o afisam pe ecran (zona sigura initiala)
        zona_sigura_initiala_pentru_afisare = zona_sigura.copy()

        imagine_augmentata = img_originala.copy()

        # desenez cu rosu leziunile deja existente (zonele interzise)
        img_cu_zone_interzise = img_originala.copy()
        for lbl in etichete_existente:
             _, x_min, y_min, x_max, y_max = lbl
             cv2.rectangle(img_cu_zone_interzise, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # cate o sarcina per clasa, cu o culoare proprie pentru cercul de marcaj
        taskuri = [
            {"set": set_noduli,  "target": random.randint(MIN_NODULI, MAX_NODULI),  "culoare": (255, 0, 0)}, # albastru pt Nodul
            {"set": set_pustule, "target": random.randint(MIN_PUSTULE, MAX_PUSTULE), "culoare": (0, 255, 0)}, # verde pt Pustula
            {"set": set_comedo,  "target": random.randint(MIN_COMEDO, MAX_COMEDO),   "culoare": (0, 0, 255)}  # albastru inchis pt Comedo
        ]

        statistici_adaugari = []

        # inserez secvential, clasa dupa clasa
        for task in taskuri:
            adaugate = 0
            incercari = 0

            while adaugate < task["target"] and incercari < 20:
                incercari += 1
                patch_original = random.choice(task["set"])

                factor_marire = random.uniform(SCALA_MARIRE_MIN, SCALA_MARIRE_MAX)
                w_nou = int(patch_original.shape[1] * factor_marire)
                h_nou = int(patch_original.shape[0] * factor_marire)

                patch = cv2.resize(patch_original, (w_nou, h_nou), interpolation=cv2.INTER_CUBIC)
                h_p, w_p = patch.shape[:2]

                # erodez zona sigura cu dimensiunea peticului => centre in care peticul incape integral
                kernel_safe = np.ones((h_p + 4, w_p + 4), np.uint8)
                zona_centre_valide = cv2.erode(zona_sigura, kernel_safe, iterations=1)

                ys, xs = np.where(zona_centre_valide > 0)
                if len(ys) == 0:
                    break

                idx_ales = random.choice(range(len(ys)))
                center_x, center_y = xs[idx_ales], ys[idx_ales]
                patch_mask = 255 * np.ones(patch.shape, patch.dtype)

                try:
                    imagine_augmentata = cv2.seamlessClone(patch, imagine_augmentata, patch_mask, (center_x, center_y), cv2.NORMAL_CLONE)
                    cv2.circle(imagine_augmentata, (center_x, center_y), max(w_p, h_p)//2 + 2, task["culoare"], 1)

                    # taiem zona folosita din zona_sigura (de lucru), dar NU din copia de afisare
                    cv2.rectangle(zona_sigura, (center_x - w_p//2 - 10, center_y - h_p//2 - 10),
                                  (center_x + w_p//2 + 10, center_y + h_p//2 + 10), 0, -1)
                    adaugate += 1
                except cv2.error:
                    pass

            statistici_adaugari.append(adaugate)

        # folosesc copia mastii pentru afisare (zona sigura initiala, neatinsa)
        zona_sigura_bgr = cv2.cvtColor(zona_sigura_initiala_pentru_afisare, cv2.COLOR_GRAY2BGR)

        # lipesc cele 3 panouri pe orizontala
        combined = np.hstack((img_cu_zone_interzise, zona_sigura_bgr, imagine_augmentata))

        h, w = combined.shape[:2]
        max_height = 600
        if h > max_height:
            scale = max_height / h
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            current_idx = (current_idx + 1) % total_images
        elif key == ord('a'):
            current_idx = (current_idx - 1) % total_images

    cv2.destroyAllWindows()

if __name__ == "__main__":
    vizualizeaza_augmentarea(DATASET_PATH)