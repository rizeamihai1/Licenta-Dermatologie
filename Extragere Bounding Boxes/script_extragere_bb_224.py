"""
Extrage crop-uri 1:1 cu context 20% din toate bbox-urile YOLO.

Structura input:
    dataset/
        train/  images/  +  labels/
        val/    images/  +  labels/
        test/   images/  +  labels/

Structura output:
    crops/
        train/
            comedo/   *.jpg
            nodul/    *.jpg
            papule/   *.jpg
            pustule/  *.jpg
        val/   (identic)
        test/  (identic)

Utilizare:
    python extract_crops.py --dataset ./dataset --output ./crops
    python extract_crops.py --dataset ./dataset --output ./crops --context 0.20 --size 224
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

CLASS_NAMES = {
    0: "comedo",
    1: "nodul",
    2: "papule",
    3: "pustule",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]

def extract_split(images_dir: Path, labels_dir: Path, output_dir: Path,
                  context: float, size: int) -> dict:
    """
    Proceseaza un singur split (train / val / test).
    Returneaza dict cu nr de crop-uri per clasa.
    """
    # Cream sub-folderele pentru fiecare clasa
    for cls_name in CLASS_NAMES.values():
        (output_dir / cls_name).mkdir(parents=True, exist_ok=True)

    counts   = {name: 0 for name in CLASS_NAMES.values()}
    skipped  = 0
    no_image = 0

    label_files = sorted(labels_dir.glob("*.txt"))

    for label_path in label_files:
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / (label_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            no_image += 1
            continue

        img    = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_h, img_w = img_np.shape[:2]

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                skipped += 1
                continue

            cls_id = int(parts[0])
            if cls_id not in CLASS_NAMES:
                skipped += 1
                continue

            cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])

            # Coordonate pixel ale bbox-ului original
            cx = cx_n * img_w
            cy = cy_n * img_h
            bw = bw_n * img_w
            bh = bh_n * img_h

            # Pasul 1: adaugam 20% context pe fiecare latura
            pad_x = bw * context
            pad_y = bh * context

            x1 = cx - bw / 2 - pad_x
            y1 = cy - bh / 2 - pad_y
            x2 = cx + bw / 2 + pad_x
            y2 = cy + bh / 2 + pad_y

            # Pasul 2: facem fereastra patrat
            # Luam latura maxima si expandam simetric in jurul centrului
            crop_w  = x2 - x1
            crop_h  = y2 - y1
            side    = max(crop_w, crop_h)

            cx_crop = (x1 + x2) / 2
            cy_crop = (y1 + y2) / 2

            x1s = cx_crop - side / 2
            y1s = cy_crop - side / 2
            x2s = cx_crop + side / 2
            y2s = cy_crop + side / 2

            # Pasul 3: calculam cat iese in afara imaginii
            pad_left   = max(0.0, -x1s)
            pad_top    = max(0.0, -y1s)
            pad_right  = max(0.0,  x2s - img_w)
            pad_bottom = max(0.0,  y2s - img_h)
            
            # clip sa ramana in imagine
            x1_clip = int(max(0,     round(x1s)))
            y1_clip = int(max(0,     round(y1s)))
            x2_clip = int(min(img_w, round(x2s)))
            y2_clip = int(min(img_h, round(y2s)))

            if x2_clip <= x1_clip or y2_clip <= y1_clip:
                skipped += 1
                continue

            crop_np = img_np[y1_clip:y2_clip, x1_clip:x2_clip].copy()

            # Pasul 4: adaug pixeli negrii daca bb a iesit in afara imaginii
            pl = int(round(pad_left))
            pt = int(round(pad_top))
            pr = int(round(pad_right))
            pb = int(round(pad_bottom))

            if pl > 0 or pt > 0 or pr > 0 or pb > 0:
                crop_np = np.pad(
                    crop_np,
                    ((pt, pb), (pl, pr), (0, 0)),
                    mode="constant",
                    constant_values=0,    # negru
                )

            # Pasul 5: resize la 224×224 cu BICUBIC
            crop_pil = Image.fromarray(crop_np).resize(
                (size, size), Image.BICUBIC
            )

            cls_name = CLASS_NAMES[cls_id]
            out_name = f"{label_path.stem}_{i:04d}.jpg"
            crop_pil.save(output_dir / cls_name / out_name, quality=95)
            counts[cls_name] += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extrage crop-uri 1:1 pentru EfficientNet"
    )
    parser.add_argument("--dataset", required=True,
                        help="Root dataset-ului (conține train/ val/ test/)")
    parser.add_argument("--output",  required=True,
                        help="Folderul de output pentru crop-uri")
    parser.add_argument("--context", type=float, default=0.20,
                        help="Context relativ adaugat pe fiecare latura (default: 0.20 = 20%%)")
    parser.add_argument("--size",    type=int,   default=224,
                        help="Dimensiunea finala a crop-ului in pixeli (default: 224)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_root  = Path(args.output)

    print("=" * 60)
    print("  EXTRAGERE CROP-URI PENTRU EFFICIENTNET")
    print(f"  Dataset : {dataset_root.resolve()}")
    print(f"  Output  : {output_root.resolve()}")
    print(f"  Context : {args.context * 100:.0f}% pe fiecare latura")
    print(f"  Size    : {args.size}×{args.size} px")
    print("=" * 60)

    grand_total = 0
    all_counts  = {}

    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        output_dir = output_root  / split

        if not images_dir.exists() or not labels_dir.exists():
            print(f"\n  [SKIP] {split} — nu am găsit images/ sau labels/")
            continue

        print(f"\n  [{split.upper()}]")
        counts = extract_split(images_dir, labels_dir, output_dir,
                               context=args.context, size=args.size)

        split_total = sum(counts.values())
        grand_total += split_total
        all_counts[split] = counts

        for cls_name, n in counts.items():
            print(f"    {cls_name:12s}: {n:>5} crop-uri")
        print(f"    {'TOTAL':12s}: {split_total:>5}")

    print("\n" + "=" * 60)
    print("  SUMAR FINAL")
    print("=" * 60)

    header = f"  {'Clasa':12s}" + "".join(f"  {s:>8}" for s in all_counts) + f"  {'TOTAL':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cls_name in CLASS_NAMES.values():
        row = f"  {cls_name:12s}"
        cls_total = 0
        for split, counts in all_counts.items():
            n = counts.get(cls_name, 0)
            row += f"  {n:>8}"
            cls_total += n
        row += f"  {cls_total:>8}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    totals_row = f"  {'TOTAL':12s}"
    for split, counts in all_counts.items():
        totals_row += f"  {sum(counts.values()):>8}"
    totals_row += f"  {grand_total:>8}"
    print(totals_row)
    print("=" * 60)


    print(f"\n  Crop-urile sunt în: {output_root.resolve()}\n")


if __name__ == "__main__":
    main()