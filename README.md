# Bank Marketing Classification

Projekat za predviđanje pretplate klijenata na bankovni rok-deposit na osnovu demografskih, profesionalnih i promotivnih karakteristika.

Cilj je izgraditi modele mašinskog učenja koji maksimizuju **F2 skor** — metriku koja Recall-u daje 4× veću težinu od Precision-a, što odgovara poslovnoj logici: propušteni klijent koji bi se pretplatio (FN) skuplji je od nepotrebnog poziva (FP).

---

## Struktura projekta

```
BankMarketingClassification/
├── data/
│   └── bank-additional-full.csv      # Sirovi dataset (41.188 uzoraka)
├── dokumentacija/
│   └── BankMarketingClassification - Tijana Marković.pdf  # Projektni izveštaj
├── src/
│   ├── data_preparation.py           # Učitavanje, čišćenje, preprocessing, split
│   ├── eda.py                        # Eksplorativna analiza podataka
│   ├── model_training.py             # Treniranje 4 modela (untuned, dva eksperimenta)
│   ├── hyperparameter_tunning.py     # GridSearchCV sa F2 scorerom
│   ├── model_evaluation.py           # Evaluacija na test skupu, threshold tuning, baseline
│   └── feature_selection.py          # Feature importance i poređenje svi vs top-N atributa
├── models/                           # Sačuvani trenirani modeli (.joblib) — gitignored
├── results/
│   ├── figures/                      # Grafikoni (ROC, confusion matrice, threshold, EDA...)
│   └── *.csv, *.txt                  # Metrike, poređenja, sažetak evaluacije
├── app.py                            # Streamlit korisnički interfejs
├── .streamlit/
│   └── config.toml                   # Tema i podešavanja Streamlit-a
└── .gitignore
```

---

## Dataset

**Bank Marketing (Additional Full)** — 41.188 uzoraka, 20 ulaznih atributa + ciljna promenljiva `y`.

Dataset je **nebalansiran**: ~88.7% klijenata nije prihvatilo ponudu (`no`), ~11.3% jeste (`yes`).

### Uklonjeni atributi

| Atribut | Razlog uklanjanja |
|---|---|
| `duration` | Data leakage — poznat tek nakon završetka poziva |
| `emp.var.rate` | Visoka korelacija sa `euribor3m` (r = 0.97) |
| `nr.employed` | Visoka korelacija sa `euribor3m` (r = 0.95) |

`euribor3m` je zadržan kao predstavnik ekonomske grupe — kvantitativno potvrđeno da uklanjanje smanjuje F2 skor (LR: 0.539 → 0.479).

---

## Pipeline

### 1. Eksplorativna analiza (`src/eda.py`)
- Distribucija ciljne promenljive i kategorijskih atributa
- Stopa pretplate po zanimanju, mesecu, tipu kontakta, ishodu prethodne kampanje
- Korelaciona matrica numeričkih atributa (7×7, bez redundantnih ekonomskih)
- Posebna analiza ekonomskih atributa (5×5 korelaciona matrica — opravdanje za uklanjanje)
- Analiza outliera i `pdays=999` vrednosti

### 2. Preprocessing (`src/data_preparation.py`)
- `prepare_dataframe()` — čišćenje, duplikati, uklanjanje `duration`
- `get_data_and_preprocessor()` — uklanjanje redundantnih ekonomskih atributa, `ColumnTransformer` (OneHotEncoder za kategorijske, passthrough za numeričke)
- `unknown` vrednosti tretirane kao validna kategorija (informativne, ne nedostajuće)
- Split: **70% train / 15% validation / 15% test**, stratifikovan, `random_state=42`

### 3. Treniranje modela (`src/model_training.py`)
- `ImbPipeline`: preprocessor → SMOTE → classifier
- `class_weight="balanced"` za LR, DT, RF (GB ne podržava)
- Dva eksperimenta: **svi atributi** vs **bez ekonomskih atributa**
- Metrika selekcije: `val_f2_yes`
- Cross-validation: `StratifiedKFold(n_splits=5)`

### 4. Hyperparameter tuning (`src/hyperparameter_tunning.py`)
- `GridSearchCV` sa F2 scorerom za sva 4 modela
- RF: `n_jobs=1` unutar klasifikatora (rešen nested parallelism)
- Čuva `best_tuned_model.joblib` (ceo pipeline, koristi ga `app.py`)

### 5. Evaluacija (`src/model_evaluation.py`)
- Evaluacija svih tuned modela na test skupu (default prag 0.5)
- Baseline poređenje: `DummyClassifier(strategy='stratified')`
- **Threshold tuning**: optimalni prag biran na validation skupu (max F2), primenjen jednom na test skupu
- ROC krive, Precision-Recall krive, confusion matrice, threshold krive

### 6. Feature selection (`src/feature_selection.py`)
- Feature importance za najbolji model (Gini / apsolutni koeficijenti)
- Poređenje performansi: svi atributi vs top-5/10/15 sirovih atributa

---

## Rezultati

| Model | F2 (val, untuned) | F2 (val, tuned) | F2 (test) | Recall (test) |
|---|---|---|---|---|
| **Logistic Regression** | 0.539 | 0.541 | **0.574** | **0.691** |
| Decision Tree | 0.459 | 0.493 | 0.541 | 0.597 |
| Random Forest | 0.310 | 0.508 | 0.540 | 0.522 |
| Gradient Boosting | 0.369 | 0.468 | 0.495 | 0.519 |

**Baseline (DummyClassifier):** F2 ≈ 0.110

**Poboljšanje najboljeg modela nad baseline-om:** +0.464 F2

Nakon threshold tuninga (prag 0.52): F2 = **0.577**, Recall = 0.677

---

## Kako pokrenuti

### Preduslovi

- Python ≥ 3.9
- Instalirati zavisnosti:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib streamlit
```

### Redosled izvršavanja

```bash
# 1. Priprema podataka (generiše models/preprocessor.joblib)
python src/data_preparation.py

# 2. Eksplorativna analiza (generiše results/figures/EDA grafikone)
python src/eda.py

# 3. Treniranje modela — untuned, oba eksperimenta
#    Generiše: models/data_splits.joblib, models/best_model.joblib,
#              results/all_features_validation_results.csv
python src/model_training.py

# 4. Hyperparameter tuning (GridSearchCV — može trajati nekoliko minuta)
#    Generiše: models/best_tuned_model.joblib,
#              results/tuned_validation_results.csv
python src/hyperparameter_tunning.py

# 5. Evaluacija na test skupu
#    Generiše: results/test_results.csv, results/threshold_tuning_results.csv,
#              results/evaluation_summary.txt, results/figures/...
python src/model_evaluation.py

# 6. Feature selection analiza
#    Generiše: results/feature_importance.csv,
#              results/feature_selection_comparison.csv
python src/feature_selection.py

# 7. Pokretanje Streamlit UI-ja
streamlit run app.py
```

> **Napomena:** Skripte se moraju pokretati iz root foldera projekta (`BankMarketingClassification/`), ne iz `src/` foldera.

> **Napomena:** `models/` folder nije uključen u repozitorijum (gitignored). Neophodno je pokrenuti skripte redosledom navedenim iznad kako bi se generisali `.joblib` fajlovi pre pokretanja `app.py`.

---

## Streamlit UI

```bash
streamlit run app.py
```

Interaktivni interfejs za predikciju pretplate novog klijenta. Unose se podaci o klijentu (lični, finansijski, kampanja, ekonomski pokazatelji) i dobija se:

- YES/NO predikcija
- Verovatnoća pretplate (%) sa progress bar-om
- Prag odluke: **0.52** (optimizovan na validation skupu po F2 kriterijumu)

---

## Tehnologije

| Paket | Namena |
|---|---|
| `pandas`, `numpy` | Manipulacija podacima |
| `scikit-learn` | Modeli, enkodiranje, metrike, GridSearchCV |
| `imbalanced-learn` | SMOTE, ImbPipeline |
| `matplotlib`, `seaborn` | Vizualizacija |
| `joblib` | Serijalizacija modela |
| `streamlit` | Korisnički interfejs |