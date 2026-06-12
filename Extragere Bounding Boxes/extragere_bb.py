import cv2
import os
from pathlib import Path

DATASET_PATH = "."  # 'unde este datasetul -> trebuie sa fie structura de tip ./train/images si ./train/labels
OUTPUT_FOLDER = "./patches_minoritare_comedo"  
TARGET_CLASS_ID = 0  # ID-ul clasei

# 0 - comedo, 1 - nodul , 2 - papule , 3 - pustule 
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def extrage_bounding_boxes(dataset_path, output_dir, target_class):
    train_img_dir = Path(dataset_path) / "train" / "images"
    train_lbl_dir = Path(dataset_path) / "train" / "labels"
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in train_img_dir.iterdir() 
        if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
    ])

    total_extrase = 0

    print(f"Incepem extragerea clasei {target_class} in folderul '{output_dir}'...")

    for img_path in image_paths:
        txt_path = train_lbl_dir / (img_path.stem + ".txt") # fisierul de labels
        
        if not txt_path.exists():
            continue
            
        # Citim imaginea o singură dată dacă are etichete
        img = None 
        
        with open(txt_path, 'r') as f:
            linii = f.readlines()
            
            contor_leziune_imagine = 0
            for linie in linii:
                parts = linie.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    if class_id == target_class:
                        if img is None:
                            img = cv2.imread(str(img_path))
                            if img is None:
                                break
                                
                        h_img, w_img = img.shape[:2]
                        
                        # Coordonate YOLO
                        cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
                        
                        cx, cy = int(cx_norm * w_img), int(cy_norm * h_img)
                        bw, bh = int(w_norm * w_img), int(h_norm * h_img)
                        
                        x_min = max(0, int(cx - bw / 2))
                        y_min = max(0, int(cy - bh / 2))
                        x_max = min(w_img, int(cx + bw / 2))
                        y_max = min(h_img, int(cy + bh / 2))
                        
                        # decupam efectiv bucata din imagine (Crop)
                        patch = img[y_min:y_max, x_min:x_max]
                        
                        if patch.shape[0] > 0 and patch.shape[1] > 0:
                            contor_leziune_imagine += 1
                            total_extrase += 1
                            
                            nume_patch = f"{img_path.stem}_patch_{contor_leziune_imagine}.jpg"
                            cale_salvare = output_path / nume_patch
                            
                            cv2.imwrite(str(cale_salvare), patch)

    print("-" * 40)
    print(f"Extragere completa! Au fost salvate {total_extrase} decupaje în folderul '{output_dir}'.")

if __name__ == "__main__":
    extrage_bounding_boxes(DATASET_PATH, OUTPUT_FOLDER, TARGET_CLASS_ID)