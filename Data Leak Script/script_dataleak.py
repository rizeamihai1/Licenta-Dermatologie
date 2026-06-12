import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import scipy.stats

# ──────────────────────────────────────────────
# CONFIGURARE
# ──────────────────────────────────────────────
DIR_IMAGINI   = './images'
DIR_LABELS    = './labels'
DIR_OUTPUT    = './Dataset_Curat_YOLO_Final_v2'

SPLIT_RATIO   = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# prag pt cosine similarity
SIMILARITY_THRESHOLD = 0.9


for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DIR_OUTPUT, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DIR_OUTPUT, split, 'labels'), exist_ok=True)


# pas 1:
# folosesc ResNet18 caruia ii scot ultimul layer (cel de clasificare)
# si pe fiecare imagine (dupa ce o normalizez) o trec prin model pentru a scoate un vector de trasaturi (512 canale)
print("=" * 60)
print("Pas 1 — ResNet18:")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device).eval() # fara ultimul strat de clasificare

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

imagini = [f for f in os.listdir(DIR_IMAGINI)
           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"\n   S-au gasit {len(imagini)} imagini. Extragem vectorii de trasaturi...\n")

features_dict = {}
for img_name in tqdm(imagini, desc="   Extragere trasaturi"):
    img_path = os.path.join(DIR_IMAGINI, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(t).cpu().numpy().flatten()
        features_dict[img_name] = vec
    except Exception as e:
        print(f"   [WARN] Skipping {img_name}: {e}")

valid_images = list(features_dict.keys())
vectors = np.array(list(features_dict.values()))


# pas 2:
# pentru fiecare imagine, calculez similaritatea cosinus cu reprezentantii grupurilor deja formate
# daca nu exista niciun grup sau daca similaritatea cu toti reprezentantii este sub prag, atunci imaginea devine reprezentantul unui nou grup
# altfel, imaginea se adauga in grupul reprezentat de cel mai apropiat reprezentant (daca depaseste pragul)

# si folosesc grupuri cu reprezentanti pt ca daca: A este similar cu B, B cu C => nu garanteaza ca si A cu C au prag de peste 0.9
print("\n" + "=" * 60)
print("Pas 2 — Crearea de grupuri cu reprezentanti...")
print("=" * 60)

sim_matrix = cosine_similarity(vectors)  # shape (N, N)

n = len(valid_images)
group_of   = [-1] * n
rep_of     = []
groups_idx = []

for i in tqdm(range(n), desc="   Grupare"):
    assigned = False
    for g_idx, rep in enumerate(rep_of):
        if sim_matrix[i][rep] >= SIMILARITY_THRESHOLD:
            groups_idx[g_idx].append(i)
            group_of[i] = g_idx
            assigned = True
            break
    if not assigned:
        g_idx = len(rep_of)
        rep_of.append(i)
        groups_idx.append([i])
        group_of[i] = g_idx

groups = [[valid_images[idx] for idx in g] for g in groups_idx]

print(f"\n    {len(valid_images)} imagini -> {len(groups)} grupuri")

# varianta 2:
# tot cu Matrice de similaritate dar in loc sa creez grupuri pe baza reprezentantilor, grupul este o componenta conexa
# 
# # from scipy.sparse.csgraph import connected_components
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# sim_matrix = cosine_similarity(vectors)
# adj_matrix = (sim_matrix >= SIMILARITY_THRESHOLD).astype(int)
# n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
# groups_idx = [[] for _ in range(n_components)]
# for img_idx, label in enumerate(labels):
#     groups_idx[label].append(img_idx)
# groups = [[valid_images[idx] for idx in g] for g in groups_idx]

# Raman cu prima varianta, le grupeaza mai bine (La a 2 a este problema de inlanturare si se formeaza grupuri uriase, care nu rezolva problema)

# pas 3: 
# pentru fiecare imagine, citesc clasele din fisierul .txt corespunzator => 
# vad cate bounding boxuri sunt in fiecare imagine (si din ce clase)
# pentru a putea creea o distributie cat mai echilibrata intre cele 3 splituri
print("\n" + "=" * 60)
print("Pas 3 — Citirea etichetelor YOLO pt a vedea distributia de clase...")
print("=" * 60)

# functie auxiliara:
# pentru o imagine data, citeste fisierul .txt corespunzator si returneaza lista de clase (fara coordonate)
def get_classes_for_image(img_name):
    base = os.path.splitext(img_name)[0]
    txt_path = os.path.join(DIR_LABELS, f"{base}.txt")
    classes = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        classes.append(int(parts[0]))
                    except ValueError:
                        pass
    return classes

# pentru fiecare grup, construiesc un counter al claselor prezente in imaginile din grup
# exemplu: [0: 3, 2: 1] inseamna ca in grup sunt 3 obiecte din clasa 0 si 1 obiect din clasa 2
group_class_vectors = []
for g in groups:
    c = Counter()
    for img in g:
        c.update(get_classes_for_image(img))
    group_class_vectors.append(c)

all_classes = sorted(set(cls for c in group_class_vectors for cls in c))
print(f"   S-au gasit {len(all_classes)} clase distincte: {all_classes}")


def histogram_vector(counter, classes):
    # Transforma un Counter intr-un vector numpy ordonat dupa clase.
    return np.array([counter.get(c, 0) for c in all_classes], dtype=float)


# ══════════════════════════════════════════════
# Pas 4 — Split-ul greedy
# ══════════════════════════════════════════════
# Strategie:
#   Pentru fiecare grup (cel mai mare primul), se evalueaza atribuirea lui la fiecare split.
#   Scor = α · (capacitate_folosita_dupa / capacitate_tinta) ->  penalizeaza supraincarcarea
#        + β · KL(distributie_clase_split_dupa ‖ distributie_clase_globala)
#
#   Scorul bazat pe "fractiunea din capacitatea tinta folosita" inseamna:
#     - Un split deja PLIN are scor >> 1.0 => nu mai este ales niciodata.
#     - Un split aproape gol are scor aproape de 0 => este preferat.
#
# Si se face split al imaginilor (nu bb) dupa 70/15/15


print("\n" + "=" * 60)
print("Pas 4 — Crearea splitului sa fie uniform...")
print("=" * 60)

ALPHA = 1.0
BETA  = 2.0

total_images  = sum(len(g) for g in groups)
target_counts = {s: total_images * r for s, r in SPLIT_RATIO.items()}

# distributia globala a claselor
global_hist = sum(
    (histogram_vector(c, all_classes) for c in group_class_vectors),
    np.zeros(len(all_classes))
)
global_prob = global_hist / (global_hist.sum() + 1e-9)

# sortez grupurile descrescator dupa numarul de imagini, ca sa aloc mai intai grupurile mari (care sunt mai greu de plasat)
order = sorted(range(len(groups)), key=lambda i: len(groups[i]), reverse=True)

split_names       = ['train', 'val', 'test']
assigned_splits   = {}
split_img_counts  = {s: 0                          for s in split_names}
split_class_hists = {s: np.zeros(len(all_classes)) for s in split_names}

def kl_divergence(p, q):
    p = np.array(p, dtype=float) + 1e-9
    q = np.array(q, dtype=float) + 1e-9
    p /= p.sum()
    q /= q.sum()
    return float(scipy.stats.entropy(p, q))

for gi in tqdm(order, desc="   Atribuire grupuri"):
    g_imgs  = groups[gi]
    g_hist  = histogram_vector(group_class_vectors[gi], all_classes)
    g_size  = len(g_imgs)

    best_split = None
    best_score = float('inf')

    for split in split_names:
        hyp_count = split_img_counts[split] + g_size
        hyp_hist  = split_class_hists[split] + g_hist

        # cat de plin ar fi splitul daca as adauga grupul asta (comparat cu tinta lui)
        # daca valoarea da peste 1 => este peste si il penalizez
        fill_ratio = hyp_count / target_counts[split]

        kl = kl_divergence(hyp_hist, global_prob * hyp_count)

        score = ALPHA * fill_ratio + BETA * kl

        if score < best_score:
            best_score = score
            best_split = split

    assigned_splits[gi]             = best_split
    split_img_counts[best_split]   += g_size
    split_class_hists[best_split]  += g_hist


# ══════════════════════════════════════════════
# Pas 5 — Raport
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pas 5 — Distributia claselor pe splituri")
print("=" * 60)

print(f"\n  {'Split':<8} {'Imagini':>8}  {'Tinta':>8}  {'Real%':>8}  "
      f"{'Tinta%':>8}  {'KL vs global':>14}")

for split in split_names:
    count  = split_img_counts[split]
    actual = count / total_images * 100
    target = SPLIT_RATIO[split] * 100
    kl     = kl_divergence(split_class_hists[split],
                            global_prob * count)
    print(f"  {split:<8} {count:>8}  {int(target_counts[split]):>8}  "
          f"{actual:>7.2f}%  {target:>7.2f}%  {kl:>14.6f}")

# Construieste maparea index grup → nume split
group_to_split = {gi: assigned_splits[gi] for gi in range(len(groups))}

# Distributia claselor per split
if all_classes:
    print(f"\n  Distributia claselor per split (numar de adnotari):")
    header = f"  {'Clasa':<8}" + "".join(f"  {s:>8}" for s in split_names) + f"  {'Global':>8}"
    print(header)
    for cls in all_classes:
        cls_idx = all_classes.index(cls)
        vals = [int(split_class_hists[s][cls_idx]) for s in split_names]
        total = sum(vals)
        row = f"  {cls:<8}" + "".join(f"  {v:>8}" for v in vals) + f"  {total:>8}"
        print(row)


# ══════════════════════════════════════════════
# Pas 6 — De copiat fisierele
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pas 6 — De copiat fisierele")
print("=" * 60)

copied_complete  = 0
copied_no_label  = 0

for gi, split in group_to_split.items():
    for img_name in groups[gi]:
        base    = os.path.splitext(img_name)[0]
        src_img = os.path.join(DIR_IMAGINI, img_name)
        src_txt = os.path.join(DIR_LABELS, f"{base}.txt")
        dst_img = os.path.join(DIR_OUTPUT, split, 'images', img_name)
        dst_txt = os.path.join(DIR_OUTPUT, split, 'labels', f"{base}.txt")

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)
            copied_complete += 1
        else:
            open(dst_txt, 'w').close()
            copied_no_label += 1



# ══════════════════════════════════════════════
# Pas 7 — Check dupa numele fisierelor sa nu fii ramas vreo poza cu acelasi nume in 2 splituri diferite
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pas 7 — Check dupa nume")
print("=" * 60)

split_image_sets = {}
for split in split_names:
    img_dir = os.path.join(DIR_OUTPUT, split, 'images')
    split_image_sets[split] = set(os.listdir(img_dir))

duplicate_found = False
pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
for s1, s2 in pairs:
    overlap = split_image_sets[s1] & split_image_sets[s2]
    if overlap:
        print(f" {s1} si {s2} au {len(overlap)} poze comune")
        duplicate_found = True
    else:
        print(f" Nu exista poze comune intre {s1} si {s2}")
