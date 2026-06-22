from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preparation import (
    DATA_PATH,
    RESULTS_DIR,
    FIGURES_DIR,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ECONOMIC_FEATURES,
    prepare_dataframe,
)

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def basic_info(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("OSNOVNE INFORMACIJE O DATASETU")
    print("=" * 60)

    print("\nPrvih nekoliko redova")
    print(dataset.head())

    print("\nDimenzije skupa")
    print(dataset.shape)

    print("\nTipovi podataka")
    print(dataset.dtypes)

    print("\nOpisna statistika numerickih atributa")
    numeric_existing = [col for col in NUMERIC_FEATURES if col in dataset.columns]
    print(dataset[numeric_existing].describe().round(2))


def missing_and_unknown_analysis(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("NEDOSTAJUCE VREDNOSTI I UNKNOWN")
    print("=" * 60)

    print("\nNaN vrednosti")
    print(dataset.isna().sum())

    print("\nUnknown vrednosti po kategorijskim atributima")
    for col in CATEGORICAL_FEATURES:
        if col in dataset.columns:
            count = (dataset[col] == "unknown").sum()
            pct = count / len(dataset) * 100
            print(f"{col}: {count} ({pct:.2f}%)")

    print("\nZakljucak:")
    print("Unknown se ne tretira kao anomalija, vec kao validna kategorijska vrednost.")


def target_distribution(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DISTRIBUCIJA CILJNE PROMENLJIVE")
    print("=" * 60)

    counts = dataset["y"].value_counts()
    pct = dataset["y"].value_counts(normalize=True) * 100

    print(counts)
    print("\nProcenti")
    print(pct.round(2))

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Distribucija ciljne promenljive")
    plt.xlabel("y")
    plt.ylabel("Broj klijenata")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def categorical_vs_target(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("ODNOS KATEGORIJSKIH ATRIBUTA I CILJNE PROMENLJIVE")
    print("=" * 60)

    for col in CATEGORICAL_FEATURES:
        if col not in dataset.columns:
            continue

        print("\nAtribut:", col)
        table = pd.crosstab(dataset[col], dataset["y"], normalize="index").round(3) * 100
        print(table)

        yes_rate = (
            dataset.groupby(col)["y"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))
        yes_rate.plot(kind="bar")
        plt.title(f"Procenat pretplate po atributu: {col}")
        plt.xlabel(col)
        plt.ylabel("% yes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col}_vs_target.png", dpi=150)
        plt.close()


def key_visualizations(dataset: pd.DataFrame):
    """
    Posebni grafici za atribute koji su najznacajniji za interpretaciju:
    contact, education, campaign, poutcome, month.
    Ovo direktno pokriva zahteve iz specifikacije projekta.
    """
    key_categorical = ["contact", "education", "poutcome", "month", "job"]

    for col in key_categorical:
        if col not in dataset.columns:
            continue

        yes_rate = (
            dataset.groupby(col)["y"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))
        yes_rate.plot(kind="bar")
        plt.title(f"Pretplata po atributu: {col}")
        plt.xlabel(col)
        plt.ylabel("% klijenata sa y=yes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"key_{col}_subscription_rate.png", dpi=150)
        plt.close()

    # campaign je numericki atribut, pa ga prikazujemo posebno.
    if "campaign" in dataset.columns:
        campaign_rate = (
            dataset.groupby("campaign")["y"]
            .apply(lambda x: (x == "yes").mean() * 100)
        )

        # Zbog ekstremnih vrednosti prikazujemo prvih 15 vrednosti kampanje.
        campaign_rate = campaign_rate[campaign_rate.index <= 15]

        plt.figure(figsize=(9, 5))
        campaign_rate.plot(kind="bar")
        plt.title("Pretplata po broju kontakata u kampanji")
        plt.xlabel("campaign")
        plt.ylabel("% yes")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "key_campaign_subscription_rate.png", dpi=150)
        plt.close()


def numeric_vs_target(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("NUMERICKI ATRIBUTI I CILJNA PROMENLJIVA")
    print("=" * 60)

    numeric_existing = [col for col in NUMERIC_FEATURES if col in dataset.columns]

    for col in numeric_existing:
        print("\nAtribut:", col)
        print(dataset.groupby("y")[col].describe().round(2))

    # Boxplotovi za numericke atribute
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(numeric_existing):
        sns.boxplot(data=dataset, x="y", y=col, ax=axes[i])
        axes[i].set_title(col)

    for j in range(len(numeric_existing), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Numericki atributi po ciljnoj promenljivoj", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_boxplots_by_target.png", dpi=150)
    plt.close()


def correlation_analysis(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KORELACIONA ANALIZA NUMERICKIH ATRIBUTA")
    print("=" * 60)

    numeric_existing = [col for col in NUMERIC_FEATURES if col in dataset.columns]

    corr = dataset[numeric_existing].corr()

    print("\nKorelaciona matrica")
    print(corr.round(3))

    print("\nJake korelacije |r| > 0.7")
    found = False
    for i in range(len(numeric_existing)):
        for j in range(i + 1, len(numeric_existing)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"{numeric_existing[i]} <-> {numeric_existing[j]}: {r:.3f}")
                found = True

    if not found:
        print("Nema jakih korelacija.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Korelaciona matrica numerickih atributa")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def economic_attributes_analysis(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("POSEBNA ANALIZA EKONOMSKIH ATRIBUTA")
    print("=" * 60)

    existing_economic = [col for col in ECONOMIC_FEATURES if col in dataset.columns]

    print("\nEkonomski atributi:")
    print(existing_economic)

    print("\nOpis po ciljnoj promenljivoj")
    for col in existing_economic:
        print("\nAtribut:", col)
        print(dataset.groupby("y")[col].describe().round(3))

    # Boxplot ekonomskih atributa po y
    fig, axes = plt.subplots(1, len(existing_economic), figsize=(4 * len(existing_economic), 5))

    if len(existing_economic) == 1:
        axes = [axes]

    for ax, col in zip(axes, existing_economic):
        sns.boxplot(data=dataset, x="y", y=col, ax=ax)
        ax.set_title(col)

    plt.suptitle("Ekonomski atributi po ciljnoj promenljivoj")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "economic_features_by_target.png", dpi=150)
    plt.close()

    # Korelacija samo ekonomskih atributa
    econ_corr = dataset[existing_economic].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(econ_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Korelacija ekonomskih atributa")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "economic_correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nNapomena:")
    print(
        "Ako su ekonomski atributi jako korelisani, model moze koristiti samo jedan od njih kao predstavnika cele grupe.")
    print("Zato se njihov znacaj dodatno proverava eksperimentom sa i bez ekonomskih atributa u model_training.py.")


def outlier_analysis(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("ANALIZA EKSTREMNIH VREDNOSTI")
    print("=" * 60)

    numeric_existing = [col for col in NUMERIC_FEATURES if col in dataset.columns]

    for col in numeric_existing:
        q1 = dataset[col].quantile(0.25)
        q3 = dataset[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((dataset[col] < lower) | (dataset[col] > upper)).sum()
        pct = outliers / len(dataset) * 100

        print(f"{col}: {outliers} outliera ({pct:.2f}%)")

    if "pdays" in dataset.columns:
        n_minus1 = (dataset["pdays"] == -1).sum()
        pct_minus1 = n_minus1 / len(dataset) * 100
        print(f"\npdays=-1: {n_minus1} ({pct_minus1:.2f}%)")
        print("pdays=-1 oznacava da klijent nije prethodno kontaktiran (originalna vrednost 999 konvertovana u -1).")


def duration_note(original_path=DATA_PATH):
    original = pd.read_csv(original_path, delimiter=";", encoding="utf-8")

    print("\n" + "=" * 60)
    print("NAPOMENA O ATRIBUTU duration")
    print("=" * 60)

    if "duration" in original.columns:
        print("Atribut duration postoji u originalnom datasetu.")
        print("On se uklanja pre treniranja jer je poznat tek nakon telefonskog razgovora.")
        print("Ukljucivanje duration atributa dovelo bi do nerealno visokih rezultata modela.")

        print("\nOpis duration atributa po ciljnoj promenljivoj")
        print(original.groupby("y")["duration"].describe().round(2))


def generate_eda_report(dataset: pd.DataFrame):
    """Generisanje sažetog izveštaja iz EDA"""
    print("\n" + "=" * 60)
    print("EDA IZVEŠTAJ - SAŽETAK")
    print("=" * 60)

    total = len(dataset)
    yes_count = dataset['y'].value_counts().get('yes', 0)
    no_count = dataset['y'].value_counts().get('no', 0)

    print(f"\n1. Ukupan broj klijenata: {total}")
    print(f"   - Pretplatilo depozit: {yes_count} ({yes_count / total * 100:.2f}%)")
    print(f"   - Nije pretplatilo: {no_count} ({no_count / total * 100:.2f}%)")
    print(f"   - Odnos: 1:{no_count / yes_count:.1f} (neuravnotežen)")

    print("\n2. Kategorijski atributi sa najvećom stopom pretplate:")
    for col in ['poutcome', 'contact', 'education', 'job']:
        if col in dataset.columns:
            top = dataset.groupby(col)['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
            print(f"   - {col}: {top.iloc[0]:.1f}% ({top.index[0]})")

    print("\n3. Numerički atributi - proseci po pretplati:")
    for col in ['age', 'campaign', 'previous']:
        if col in dataset.columns:
            yes_mean = dataset[dataset['y'] == 'yes'][col].mean()
            no_mean = dataset[dataset['y'] == 'no'][col].mean()
            print(f"   - {col}: yes={yes_mean:.1f}, no={no_mean:.1f}")

    print("\n4. Uočene anomalije:")
    print("   - pdays=-1 označava odsustvo prethodnog kontakta (konvertovano iz 999)")
    print("   - 'unknown' vrednosti su validne kategorije")
    print("   - duration je uklonjen kao leaky feature")


if __name__ == "__main__":
    # prepare_dataframe ucitava podatke, proverava missing/unknown,
    # uklanja duplikate i uklanja duration.
    dataset = prepare_dataframe(DATA_PATH)

    basic_info(dataset)
    missing_and_unknown_analysis(dataset)
    target_distribution(dataset)
    categorical_vs_target(dataset)
    key_visualizations(dataset)
    numeric_vs_target(dataset)
    correlation_analysis(dataset)
    economic_attributes_analysis(dataset)
    outlier_analysis(dataset)
    duration_note(DATA_PATH)
    generate_eda_report(dataset)

    print("\n" + "=" * 60)
    print("EDA ZAVRSENA")
    print("Grafici su sacuvani u folderu:", FIGURES_DIR)
    print("=" * 60)