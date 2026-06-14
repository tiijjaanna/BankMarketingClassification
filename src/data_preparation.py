from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DURATION_COLUMN = "duration"
TARGET_COLUMN = "y"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    print("Ucitavanje podataka")
    dataset = pd.read_csv(path, delimiter=";", encoding="utf-8")
    print("Prvih nekoliko redova")
    print(dataset.head())
    print("\nDimenzije skupa")
    print(dataset.shape)
    return dataset


def handle_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    print("\nNedostajuce vrednosti pre ciscenja:")
    print(dataset.isna().sum())

    dataset = dataset.dropna()
    dataset = dataset.drop(columns=[DURATION_COLUMN])

    print("\nNedostajuce vrednosti nakon ciscenja:")
    print(dataset.isna().sum())

    return dataset


def remove_duplicates(dataset: pd.DataFrame) -> pd.DataFrame:
    print("\nBroj duplikata:")
    duplicate_sum = dataset.duplicated().sum()
    print(duplicate_sum)

    if duplicate_sum > 0:
        print(f"\nPronadjeno duplikata: {duplicate_sum}")
        dataset = dataset.drop_duplicates(keep="first")
        print("Duplikati obrisani")

    return dataset


def check_distribution_or_target_column(dataset: pd.DataFrame):
    print("\nDistribucija ciljne promenljive (y):")
    print(dataset[TARGET_COLUMN].value_counts())
    target_pct = dataset[TARGET_COLUMN].value_counts(normalize=True) * 100
    print(f"\nDistribucija u procentima:\n{target_pct.round(2).to_string()}")


def encoding(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = encoding_education(dataset)
    dataset = encode_binary_label(dataset)
    dataset = one_hot_encoding(dataset)
    dataset = encoding_cyclical(dataset)
    return dataset


def encoding_education(dataset: pd.DataFrame) -> pd.DataFrame:
    education_order = {
        'illiterate': 0,
        'basic.4y': 1,
        'basic.6y': 2,
        'basic.9y': 3,
        'high.school': 4,
        'professional.course': 5,
        'university.degree': 6,
        'unknown': 7,
    }

    dataset['education'] = dataset['education'].map(education_order)

    missing_edu = dataset['education'].isna().sum()
    print(f"[education] Ordinalno enkodiranje – nepokriventih vrednosti: {missing_edu}")
    print(f"  Vrednosti posle enkodiranja: {sorted(dataset['education'].unique())}\n")

    return dataset


def encode_binary_label(dataset: pd.DataFrame) -> pd.DataFrame:
    # default, housing, loan imaju 3 vrednosti: yes/no/unknown
    three_val_map = {'no': 0, 'yes': 1, 'unknown': 2}

    for col in ['default', 'housing', 'loan']:
        dataset[col] = dataset[col].map(three_val_map)
        print(f"[{col}] Label Encoding: no→0, yes→1, unknown→2 | "
              f"Raspodela: {dataset[col].value_counts().to_dict()}")

    contact_map = {'cellular': 1, 'telephone': 0}
    dataset['contact'] = dataset['contact'].map(contact_map)
    print(f"[contact] Label Encoding: telephone→0, cellular→1 | "
          f"Raspodela: {dataset['contact'].value_counts().to_dict()}\n")

    dataset['y'] = dataset['y'].map({'yes': 1, 'no': 0})
    print(f"[y] Ciljna promenljiva: no→0, yes→1 | "
          f"Raspodela: {dataset['y'].value_counts().to_dict()}\n")

    return dataset


def one_hot_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
    ohe_cols = ['job', 'marital', 'poutcome']
    dataset = pd.get_dummies(dataset, columns=ohe_cols, drop_first=True, dtype=int)

    new_ohe_cols = [c for c in dataset.columns
                    if any(c.startswith(p + '_') for p in ohe_cols)]
    print(f"[job, marital, poutcome] One-Hot Encoding")
    print(f"  Novokreirane kolone ({len(new_ohe_cols)}): {new_ohe_cols}\n")

    return dataset


def encoding_cyclical(dataset: pd.DataFrame) -> pd.DataFrame:
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    dataset['month_num'] = dataset['month'].map(month_map)
    dataset['month_sin'] = np.sin(2 * np.pi * dataset['month_num'] / 12)
    dataset['month_cos'] = np.cos(2 * np.pi * dataset['month_num'] / 12)
    dataset = dataset.drop(columns=['month', 'month_num'])
    print(f"[month] Ciklicno enkodiranje → month_sin, month_cos")

    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
    dataset['day_num'] = dataset['day_of_week'].map(day_map)
    dataset['day_sin'] = np.sin(2 * np.pi * dataset['day_num'] / 5)
    dataset['day_cos'] = np.cos(2 * np.pi * dataset['day_num'] / 5)
    dataset = dataset.drop(columns=['day_of_week', 'day_num'])
    print(f"[day_of_week] Ciklicno enkodiranje → day_sin, day_cos\n")

    return dataset


def get_prepared_data(path=DATA_PATH):
    """Pomocna funkcija – vraca X i y spremne za treniranje modela."""
    dataset = load_data(path)
    dataset = handle_missing_values(dataset)
    dataset = remove_duplicates(dataset)
    dataset = encoding(dataset)

    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    dataset = load_data()
    dataset = handle_missing_values(dataset)
    dataset = remove_duplicates(dataset)
    check_distribution_or_target_column(dataset)
    dataset = encoding(dataset)

    print(f"\nFinalni dataset: {dataset.shape[0]} redova x {dataset.shape[1]} kolona")
    print(f"Ukupno NaN vrednosti: {dataset.isna().sum().sum()}")
    print("\nKolone finalnog dataseta:")
    print(list(dataset.columns))

