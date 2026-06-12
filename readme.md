# DETECȚIA ȘI CLASIFICAREA LEZIUNILOR ACNEICE PRIN VIZIUNE ARTIFICIALĂ

Acest repository conține întregul pipeline de cercetare și dezvoltare pentru un sistem capabil să detecteze și să clasifice leziunile acneice din imagini faciale. Proiectul abordează provocări specifice analizei imaginilor medicale, precum dezechilibrul claselor, variabilitatea morfologică și riscul de scurgere a datelor (data leakage).

## Structura Proiectului

Soluția este modularizată în patru componente principale, fiecare având propriul director și propria documentație:

- **`Data Leak Script/`**
  Conține scripturile responsabile pentru pregătirea setului de date. Folosește o abordare bazată pe Deep Learning (ResNet18 + Similaritate Cosinus) pentru a grupa imaginile cu același pacient, asigurând o împărțire corectă (train/val/test) care previne complet scurgerea de date.

- **`Augmentare Script/`**
  Implementează un algoritm personalizat de augmentare offline. Sistemul extrage o mască facială binară în spațiul de culoare YCrCb și folosește tehnici de Poisson blending (Seamless Clone) pentru a insera leziuni din clasele minoritare, echilibrând astfel setul de antrenare.

- **`Antrenare Modele - Prin Google Colab/`**
  Găzduiește experimentele și scripturile de antrenament. Include antrenarea modelelor de detecție (YOLOv8m, EfficientDet-D1), studiile de ablații pentru rezolvarea dezechilibrului claselor, precum și antrenarea clasificatorului final (EfficientNet-B0).

- **`Extragere Bounding Boxes/`**
  Conține utilitarele necesare pentru a face trecerea de la modelul de detecție la cel de clasificare. Scripturile decupează leziunile detectate (sau adnotările de referință) pentru a crea setul de date necesar antrenării clasificatorului independent.

## Fluxul de Lucru (Pipeline)

1. **Curățarea și Împărțirea Datelor:** Rularea scriptului de prevenire a data leakage-ului pentru obținerea seturilor train/val/test.
   Setul de date rezultat se poate gasii pe Google Drive la urmatorul link: https://drive.google.com/drive/folders/1sFv92JaBFlQRYv0s_reEKDpyfrg2UWyW
2. **Augmentarea:** Echilibrarea setului de antrenare prin inserarea inteligentă a leziunilor subreprezentate.
3. **Detecția:** Antrenarea modelului YOLOv8m pentru a localiza leziunile pe întreaga imagine facială.
4. **Clasificarea:** Extragerea detecțiilor și trecerea lor prin modelul EfficientNet-B0 pentru o clasificare precisă a tipului de leziune.

### Exemple de Leziuni

| ![Comedo 0](imagini%20aplicatie/exemple%20leziuni/comedo%200.png) | ![Comedo 1](imagini%20aplicatie/exemple%20leziuni/comedo%201.png) |   ![Nodul](imagini%20aplicatie/exemple%20leziuni/nodul.png)   |
| :---------------------------------------------------------------: | :---------------------------------------------------------------: | :-----------------------------------------------------------: |
|  ![Nodul 2](imagini%20aplicatie/exemple%20leziuni/nodul%202.png)  |    ![Papule](imagini%20aplicatie/exemple%20leziuni/papule.png)    | ![Pustule](imagini%20aplicatie/exemple%20leziuni/pustule.png) |

### Exemple de Adnotari

### Adnotări

![Adnotare 1](imagini%20aplicatie/adnotare-1.png)

![Adnotare 2](imagini%20aplicatie/adnotare-2.png)

## Aplicatia Django se regaseste alaturi de modelele YOLO Multiclass + Sistemul Secvential compus din YOLO Singleclass si EfficientNet-B0 in `Aplicatie Django / acne_demo / models_dir `

## Rezultate și Detecții

### Comparație Detecții vs. Ground Truth (GT)

Detectii imagine 1:
![1 Detectii](imagini%20aplicatie/1%20detectii.png)
Groud Truth 1:
![1 GT](imagini%20aplicatie/1%20GT.png)

Detectii imagine 2:
![2 Detectii](imagini%20aplicatie/2%20detectii.png)
Ground Truth 2:
![2 GT](imagini%20aplicatie/2%20GT.png)

---

### Exemple Suplimentare

| ![Exemple detectii 1](imagini%20aplicatie/Exemple%20detectii%20v2.png) | ![Exemple detectii 2](imagini%20aplicatie/Exemple%20detectii%202%20v2.png) |
| :--------------------------------------------------------------------: | :------------------------------------------------------------------------: |
