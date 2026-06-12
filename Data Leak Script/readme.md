# Eliminarea Duplicatelor și Împărțirea Setului de Date

Unificarea celor trei surse de date a expus o problemă critică: sursele originale conțineau imagini pre-augmentate, același subiect (pacient) apărând de până la 10 ori prin rotații, oglindiri și translații ale imaginii de bază. Prezența imaginilor cu același subiect atât în setul de antrenare, cât și în cel de testare ar invalida complet evaluarea modelului (**Data Leakage**).

---

### Calculul Similarității Vizuale

Pentru identificarea automată a duplicatelor, am utilizat arhitectura **ResNet18** fără ultimul strat de clasificare, producând un vector de trăsături $\mathbf{f}_i \in \mathbb{R}^{512}$ pentru fiecare imagine. Similaritatea dintre două imagini se calculează matematic prin **similaritatea cosinus**:

$$\text{sim}(\mathbf{f}_i, \mathbf{f}_j) = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\|\mathbf{f}_i\|\,\|\mathbf{f}_j\|}$$

Algoritmul folosește un prag stabil de **$\tau = 0.9$** pentru a grupa imaginile considerate similare vizual.

---

### Algoritmii folositi:

#### Algoritmul 1: Extragerea vectorilor de trăsături cu ResNet18

- **Date de intrare:** Dataset $\mathcal{D} = \{I_i\}_{i=1}^N$
- **Date de ieșire:** Dicționar de trăsături $\mathcal{F} = \{I_i \mapsto \mathbf{f}_i\}$

$$
\begin{aligned}
&1. \ \mathcal{F} \leftarrow \emptyset \\
&2. \ \text{Pentru fiecare imagine } I_i \in \mathcal{D}: \\
&3. \quad \mathbf{f}_i \leftarrow \text{ResNet18}(I_i) \ (\text{vector } \in \mathbb{R}^{512}) \\
&4. \quad \mathcal{F} \leftarrow \mathcal{F} \cup \{I_i \mapsto \mathbf{f}_i\} \\
&5. \ \text{Returnează } \mathcal{F}
\end{aligned}
$$

#### Algoritmul 2: Grupare cu reprezentanți a imaginilor similare

- **Date de intrare:** Dicționar de trăsături $\mathcal{F} = \{I_i \mapsto \mathbf{f}_i\}$, prag $\tau = 0.9$
- **Date de ieșire:** Grupuri finale $\mathcal{G} = \{\mathcal{G}_1, \dots, \mathcal{G}_K\}$

$$
\begin{aligned}
&1. \ R \leftarrow \emptyset, \ \mathcal{G} \leftarrow \emptyset \\
&2. \ \text{Pentru fiecare imagine } I_i \in \mathcal{D}: \\
&3. \quad asignat \leftarrow \text{Fals} \\
&4. \quad \text{Pentru fiecare reprezentant } R_k \in R: \\
&5. \qquad sim \leftarrow \frac{\mathbf{f}_i \cdot \mathbf{f}_{R_k}}{\|\mathbf{f}_i\|\,\|\mathbf{f}_{R_k}\|} \\
&6. \qquad \text{Dacă } sim \ge \tau \text{ atunci:} \\
&7. \quad \qquad \mathcal{G}_k \leftarrow \mathcal{G}_k \cup \{I_i\} \\
&8. \quad \qquad asignat \leftarrow \text{Adevărat} \\
&9. \quad \qquad \textbf{break} \\
&10. \ \text{Dacă \textbf{nu} } asignat \text{ atunci:} \\
&11. \quad \ R \leftarrow R \cup \{I_i\} \\
&12. \quad \ \mathcal{G} \leftarrow \mathcal{G} \cup \{\{I_i\}\} \\
&13. \text{Returnează } \mathcal{G}
\end{aligned}
$$

---

### Strategia Greedy de Împărțire (Split Uniform)

Grupurile rezultate sunt sortate descrescător după numărul de imagini din fiecare și sunt atribuite iterativ subseturilor finale conform rapoartelor stabilite: **Antrenare (70%)**, **Validare (15%)** și **Testare (15%)**.

- **Logica de alocare:** La fiecare pas, grupul curent (cel mai mare rămas) este alocat acelui subset care se află cel mai departe de capacitatea sa țintă, având grijă ca distribuția claselor de leziuni din subset să rămână cât mai apropiată de distribuția globală.
- **Rezultat:** Această strategie de tip _greedy_ minimizează la maximum riscul de _data leakage_ (scurgere de date) și asigură o consistență perfectă a distribuției claselor între toate cele trei subseturi.
