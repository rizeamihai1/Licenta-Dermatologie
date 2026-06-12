# Sistem Automat de Detecție și Clasificare a Leziunilor Acneice

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
2. **Augmentarea:** Echilibrarea setului de antrenare prin inserarea inteligentă a leziunilor subreprezentate.
3. **Detecția:** Antrenarea modelului YOLOv8m pentru a localiza leziunile pe întreaga imagine facială.
4. **Clasificarea:** Extragerea detecțiilor și trecerea lor prin modelul EfficientNet-B0 pentru o clasificare precisă a tipului de leziune.
