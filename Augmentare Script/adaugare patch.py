import cv2
import numpy as np
import os
import random
import shutil
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


# 2. Iau etichetele YOLO si din forma lor normalizata le convertesc in pixeli
# pentru a putea prelucra si mai mult masca, adaugand ca "zona interzisa" si leziunile deja existente
def citeste_etichete_yolo(txt_path, w_img, h_img):
    # returneaza [class_id, x_min, y_min, x_max, y_max]
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

def pixel_to_yolo(class_id, x_min, y_min, x_max, y_max, w_img, h_img):
    # Functia aceasta converteste din pixeli inapoi in format YOLO, pentru a le baga inapoi in fisierul de labels corect
    cx = (x_min + x_max) / 2.0 / w_img
    cy = (y_min + y_max) / 2.0 / h_img
    bw = (x_max - x_min) / w_img
    bh = (y_max - y_min) / h_img
    # Clampăm la [0, 1] pentru siguranță
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def salveaza_etichete_yolo(txt_path, etichete_originale, etichete_noi, w_img, h_img):
    # pun inapoi si etichetele vechi + cele nou adaugate in fisier
    linii = []
    for lbl in etichete_originale:
        class_id, x_min, y_min, x_max, y_max = lbl
        linii.append(pixel_to_yolo(class_id, x_min, y_min, x_max, y_max, w_img, h_img))
    for lbl in etichete_noi:
        class_id, x_min, y_min, x_max, y_max = lbl
        linii.append(pixel_to_yolo(class_id, x_min, y_min, x_max, y_max, w_img, h_img))
    with open(txt_path, 'w') as f:
        f.write("\n".join(linii))

def creaza_masca_interzisa(labels, w_img, h_img, padding=10):
    forbidden = np.zeros((h_img, w_img), dtype=np.uint8)
    for lbl in labels:
        _, x_min, y_min, x_max, y_max = lbl
        cv2.rectangle(forbidden,
                      (max(0, x_min - padding), max(0, y_min - padding)),
                      (min(w_img, x_max + padding), min(h_img, y_max + padding)),
                      255, -1)
    return forbidden

# 3. Incarc setul de leziuni
# pentru fiecare clasa subreprezentata (fata de papule) citesc toate peticele (crop-urile) din folderul ei
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


# 4. Augmentarea unei singure imagini
# construiesc zona sigura (piele - leziuni existente) si inserez leziuni din seturi pana ating tinta
def augmenteaza_imagine(img_originala, etichete_existente,
                        set_noduli, set_pustule, set_comedo):
    # returneaza imaginea augmentata si lista de bbox-uri noi [class_id, x_min, y_min, x_max, y_max] (pixeli)
    h_img, w_img = img_originala.shape[:2]

    masca_piele     = build_skin_mask_dynamic(img_originala)
    masca_interzisa = creaza_masca_interzisa(etichete_existente, w_img, h_img)
    zona_sigura     = cv2.bitwise_and(masca_piele, cv2.bitwise_not(masca_interzisa))  # piele fara leziunile deja existente

    imagine_aug  = img_originala.copy()
    etichete_noi = []

    # cate o sarcina per clasa, cu un numar aleator de leziuni de inserat
    taskuri = [
        {
            "set":      set_noduli,
            "target":   random.randint(MIN_NODULI, MAX_NODULI),
            "class_id": NODUL_CLASS_ID,
        },
        {
            "set":      set_pustule,
            "target":   random.randint(MIN_PUSTULE, MAX_PUSTULE),
            "class_id": PUSTULA_CLASS_ID,
        },
        {
            "set":      set_comedo,
            "target":   random.randint(MIN_COMEDO, MAX_COMEDO),
            "class_id": COMEDO_CLASS_ID,
        }
    ]

    for task in taskuri:
        adaugate  = 0
        incercari = 0

        # cel mult 20 de incercari per clasa (pe imagini mici se poate sa nu incapa toate)
        while adaugate < task["target"] and incercari < 20:
            incercari += 1
            patch_original = random.choice(task["set"])

            # variatie de scara a peticului
            factor_marire = random.uniform(SCALA_MARIRE_MIN, SCALA_MARIRE_MAX)
            w_nou = int(patch_original.shape[1] * factor_marire)
            h_nou = int(patch_original.shape[0] * factor_marire)
            patch = cv2.resize(patch_original, (w_nou, h_nou), interpolation=cv2.INTER_CUBIC)
            h_p, w_p = patch.shape[:2]

            # erodez zona sigura cu dimensiunea peticului (+2px) => raman doar centrele in care peticul incape integral
            kernel_safe = np.ones((h_p + 4, w_p + 4), np.uint8)
            zona_centre_valide = cv2.erode(zona_sigura, kernel_safe, iterations=1)

            # toate coordonatele valide pentru centrul peticului
            ys, xs = np.where(zona_centre_valide > 0)
            if len(ys) == 0:
                break

            # aleg aleator un centru dintre cele valide
            idx_ales  = random.choice(range(len(ys)))
            center_x  = xs[idx_ales]
            center_y  = ys[idx_ales]
            patch_mask = 255 * np.ones(patch.shape, patch.dtype)

            try:
                # incerc sa inserez peticul cu seamlessClone
                imagine_aug = cv2.seamlessClone(
                    patch, imagine_aug, patch_mask,
                    (center_x, center_y), cv2.NORMAL_CLONE
                )

                # bbox-ul exact: seamlessClone centreaza peticul pe (center_x, center_y)
                x_min = center_x - w_p // 2
                y_min = center_y - h_p // 2
                x_max = center_x + w_p // 2
                y_max = center_y + h_p // 2

                # sa fie la limitele imaginii
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w_img, x_max)
                y_max = min(h_img, y_max)

                etichete_noi.append([task["class_id"], x_min, y_min, x_max, y_max])

                # blochez zona folosita (+10px) ca sa nu se suprapuna leziunile urmatoare
                cv2.rectangle(zona_sigura,
                              (center_x - w_p // 2 - 10, center_y - h_p // 2 - 10),
                              (center_x + w_p // 2 + 10, center_y + h_p // 2 + 10),
                              0, -1)
                adaugate += 1

            except cv2.error:
                pass  # seamlessClone poate esua pe margini => ignor si incerc alta pozitie

    return imagine_aug, etichete_noi


# 5. Pipeline-ul principal: parcurge imaginile de train si salveaza variantele augmentate
def ruleaza_augmentare_si_salveaza(dataset_path):
    train_img_dir = Path(dataset_path) / "train" / "images"
    train_lbl_dir = Path(dataset_path) / "train" / "labels"

    out_img_dir = Path(OUTPUT_IMAGES_DIR)
    out_lbl_dir = Path(OUTPUT_LABELS_DIR)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print("  Augmentare YOLO cu salvare")
    print(f"  Input : {train_img_dir}")
    print(f"  Output: {out_img_dir}")
    print("-" * 60)

    set_noduli  = incarca_set_leziuni(PATCHES_PATH_NODUL)
    set_pustule = incarca_set_leziuni(PATCHES_PATH_PUSTULE)
    set_comedo  = incarca_set_leziuni(PATCHES_PATH_COMEDO)
    if not set_noduli or not set_pustule or not set_comedo:
        print("[EROARE] Unul dintre seturile de leziuni este gol. Oprit.")
        return

    image_paths = sorted([
        p for p in train_img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
    ])

    if not image_paths:
        print("[EROARE] Nu am gasit imagini in directorul de antrenament.")
        return

    total_salvate = 0

    for img_path in image_paths:
        img_originala = cv2.imread(str(img_path))
        if img_originala is None:
            print(f"  [SKIP] Nu am putut citi: {img_path.name}")
            continue

        h_img, w_img = img_originala.shape[:2]
        txt_path      = train_lbl_dir / (img_path.stem + ".txt")
        etichete_orig = citeste_etichete_yolo(str(txt_path), w_img, h_img)

        print(f"\n  [{img_path.name}] - {AUGMENTARI_PER_IMAGINE} variante...")

        # generez mai multe variante augmentate per imagine
        for aug_idx in range(1, AUGMENTARI_PER_IMAGINE + 1):
            imagine_aug, etichete_noi = augmenteaza_imagine(
                img_originala, etichete_orig,
                set_noduli, set_pustule, set_comedo
            )

            # numele fisierelor de iesire
            stem_out = f"{img_path.stem}_aug{aug_idx:02d}"
            out_img_path = out_img_dir / f"{stem_out}{img_path.suffix}"
            out_lbl_path = out_lbl_dir / f"{stem_out}.txt"

            # salvez imaginea
            cv2.imwrite(str(out_img_path), imagine_aug)

            # salvez etichetele (originale + cele nou inserate)
            salveaza_etichete_yolo(
                str(out_lbl_path),
                etichete_orig,
                etichete_noi,
                w_img, h_img
            )

            n_noi = len(etichete_noi)
            n_total = len(etichete_orig) + n_noi
            print(f"    {stem_out}: +{n_noi} leziuni noi -> {n_total} bbox-uri totale")
            total_salvate += 1

    print("\n" + "-" * 60)
    print(f"  GATA! {total_salvate} imagini augmentate salvate in {out_img_dir}")
    print("-" * 60)

if __name__ == "__main__":
    ruleaza_augmentare_si_salveaza(DATASET_PATH)
