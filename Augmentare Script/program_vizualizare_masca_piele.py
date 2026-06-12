import cv2
import numpy as np
import os
from pathlib import Path

DATASET_PATH = "."
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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


# 2. Script de vizualizare a mastii pielii

def visualise_skin_masks(dataset_path: str):
    train_img_dir = Path(dataset_path) / "train" / "images"

    image_paths = sorted([
        p for p in train_img_dir.iterdir() 
        if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
    ])

    current_idx = 0
    total_images = len(image_paths)
    
    window_name = "(D: Inainte | A: Inapoi | Q: Iesire)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)

    print("=== NAVIGARE ===")
    print(" D - Imaginea urmatoare")
    print(" A - Imaginea anterioara")
    print(" Q - Iesire din program\n")

    while True:
        img_path = image_paths[current_idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            current_idx = (current_idx + 1) % total_images
            continue

        mask = build_skin_mask_dynamic(img)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack((img, mask_bgr))
        
        h, w = combined.shape[:2]
        max_height = 800
        if h > max_height:
            scale = max_height / h
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

        cv2.putText(combined, f"{current_idx + 1}/{total_images}: {img_path.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("Program oprit")
            break
        elif key == ord('d'):
            current_idx = (current_idx + 1) % total_images
        elif key == ord('a'):
            current_idx = (current_idx - 1) % total_images

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualise_skin_masks(DATASET_PATH)