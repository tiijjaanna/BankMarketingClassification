from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "y"
LEAKY_COLUMN = "duration"

# unknown se tretira kao validna kategorija i ostaje u OneHotEncoder-u.
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# emp.var.rate i nr.employed su uklonjeni: jako su korelisani sa euribor3m
# (r > 0.9 za oba), pa nose redundantnu informaciju. euribor3m je zadrzan
# kao predstavnik ekonomske grupe atributa jer je najpoznatiji i
# najinterpretabilniji u kontekstu bankarstva.
NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous",
    "cons.price.idx", "cons.conf.idx", "euribor3m",
]

# Originalna ekonomska grupa, ostavljena radi reference u eda.py
# (correlation_analysis i economic_attributes_analysis se i dalje racunaju
# na celoj grupi da bi se korelacija mogla prikazati i opravdati).
ECONOMIC_FEATURES = [
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]

CLIENT_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
]

CAMPAIGN_FEATURES = [
    "contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    print("Ucitavanje podataka")
    dataset = pd.read_csv(path, delimiter=";", encoding="utf-8")
    print("\nPrvih nekoliko redova")
    print(dataset.head())
    print("\nDimenzije skupa")
    print(dataset.shape)
    return dataset


def initial_checks(dataset: pd.DataFrame) -> None:
    print("\nNedostajuce vrednosti")
    print(dataset.isna().sum())

    print("\nBroj 'unknown' vrednosti po kategorijskim atributima")
    for col in CATEGORICAL_FEATURES:
        n = (dataset[col] == "unknown").sum()
        pct = n / len(dataset) * 100
        print(f"{col}: {n} ({pct:.2f}%)")

    print("\nNapomena: 'unknown' se tretira kao validna kategorija, ne kao anomalija.")
    print("Razlog: stopa pretplate kod 'unknown' vrednosti (npr. kod default: ~5%)")
    print("znacajno odstupa od 'no' (~13%), sto pokazuje da je 'unknown' informativna")
    print("kategorija, a ne slucajno nedostajuci podatak koji treba nadoknaditi.")


def remove_duplicates(dataset: pd.DataFrame) -> pd.DataFrame:
    print("\nBroj duplikata pre ciscenja")
    duplicate_count = dataset.duplicated().sum()
    print(duplicate_count)
    if duplicate_count > 0:
        dataset = dataset.drop_duplicates().reset_index(drop=True)
    print("\nBroj duplikata nakon ciscenja")
    print(dataset.duplicated().sum())
    return dataset


def remove_leaky_features(dataset: pd.DataFrame) -> pd.DataFrame:
    if LEAKY_COLUMN in dataset.columns:
        dataset = dataset.drop(columns=[LEAKY_COLUMN])
        print(f"\nKolona '{LEAKY_COLUMN}' je uklonjena jer predstavlja leaky feature.")
        print("Razlog: trajanje razgovora je poznato tek nakon poziva, ne pre donosenja odluke.")
    return dataset

def convert_pdays(dataset: pd.DataFrame) -> pd.DataFrame:
    """Konvertuje pdays=999 u -1 (označava odsustvo prethodnog kontakta)."""
    if "pdays" in dataset.columns:
        dataset["pdays"] = dataset["pdays"].replace(999, -1)
    return dataset

def remove_redundant_economic_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Uklanja emp.var.rate i nr.employed - jako korelisani sa euribor3m (r>0.9)."""
    redundant = [c for c in ["emp.var.rate", "nr.employed"] if c in dataset.columns]
    if redundant:
        dataset = dataset.drop(columns=redundant)
        print(f"\nKolone {redundant} su uklonjene zbog jake korelacije (r>0.9) sa euribor3m.")
        print("euribor3m je zadrzan kao predstavnik ekonomske grupe atributa.")
    return dataset


def prepare_dataframe(path=DATA_PATH) -> pd.DataFrame:
    dataset = load_data(path)
    initial_checks(dataset)
    dataset = remove_duplicates(dataset)
    dataset = remove_leaky_features(dataset)
    dataset = convert_pdays(dataset)  
    return dataset


def get_data_and_preprocessor(path=DATA_PATH, selected_features=None):
    dataset = prepare_dataframe(path)
    dataset = remove_redundant_economic_features(dataset)

    if selected_features is None:
        categorical_features = CATEGORICAL_FEATURES.copy()
        numeric_features = NUMERIC_FEATURES.copy()
    else:
        categorical_features = [c for c in CATEGORICAL_FEATURES if c in selected_features]
        numeric_features = [c for c in NUMERIC_FEATURES if c in selected_features]

    X = dataset[categorical_features + numeric_features].copy()
    y = dataset[TARGET_COLUMN].map({"no": 0, "yes": 1})

    print("\nRaspodela ciljne promenljive")
    print(y.value_counts())
    print((y.value_counts(normalize=True) * 100).round(2))

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    return X, y, preprocessor, categorical_features, numeric_features


def demonstrate_preprocessing(X, preprocessor):
    X_preprocessed = preprocessor.fit_transform(X)
    print("\nOblik X pre preprocessing-a")
    print(X.shape)
    print("\nOblik X nakon preprocessing-a")
    print(X_preprocessed.shape)
    print("\nNovi atributi nakon enkodiranja")
    print(preprocessor.get_feature_names_out())
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    print("\nPreprocessor sacuvan kao models/preprocessor.joblib")


def get_feature_names(preprocessor):
    """Dobijanje imena atributa nakon OneHotEncoder-a"""
    return preprocessor.get_feature_names_out()


def get_data_splits():
    """Vraća podeljene podatke ako postoje, ili ih kreira"""
    splits_path = MODEL_DIR / "data_splits.joblib"
    if splits_path.exists():
        print("Učitavanje postojećih podeljenih podataka...")
        return joblib.load(splits_path)
    else:
        print("Kreiranje novih podeljenih podataka...")
        from sklearn.model_selection import train_test_split
        X, y, preprocessor, cat_features, num_features = get_data_and_preprocessor()
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
        splits = (X_train, X_val, X_test, y_train, y_val, y_test)
        joblib.dump(splits, splits_path)
        print("Podaci sačuvani u models/data_splits.joblib")
        return splits


if __name__ == "__main__":
    X, y, preprocessor, categorical_features, numeric_features = get_data_and_preprocessor(DATA_PATH)
    print("\nKategorijski atributi")
    print(categorical_features)
    print("\nNumericki atributi")
    print(numeric_features)
    demonstrate_preprocessing(X, preprocessor)