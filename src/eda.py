from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

num_cols = [
    'age', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

cat_cols = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month',
    'day_of_week', 'poutcome'
]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(path, delimiter=";", encoding="utf-8")
    print(f"Ucitan dataset: {dataset.shape[0]} redova x {dataset.shape[1]} kolona")
    return dataset


def basic_info(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("OSNOVNE INFORMACIJE O DATASETU")
    print("=" * 60)

    print("\nTipovi podataka:")
    print(dataset.dtypes)

    print("\nOpisna statistika – numericki atributi:")
    print(dataset[num_cols].describe().round(2))

    print("\nBroj jedinstvenih vrednosti po kategorickim atributima:")
    for col in cat_cols:
        print(f"  {col}: {sorted(dataset[col].unique())}")


def check_missing_and_unknown(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("NEDOSTAJUCE VREDNOSTI I 'UNKNOWN'")
    print("=" * 60)

    print("\nBroj NaN vrednosti po koloni:")
    print(dataset.isna().sum())

    print("\nBroj 'unknown' vrednosti po kategorickim kolonama:")
    for col in cat_cols:
        n = (dataset[col] == 'unknown').sum()
        pct = n / len(dataset) * 100
        print(f"  {col}: {n} ({pct:.1f}%)")


def check_outliers(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EKSTREMNE VREDNOSTI (OUTLIERI) – IQR METODA")
    print("=" * 60)

    # IQR metoda: vrednosti ispod Q1-1.5*IQR ili iznad Q3+1.5*IQR su outlieri
    for col in num_cols:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((dataset[col] < lower) | (dataset[col] > upper)).sum()
        pct = n_out / len(dataset) * 100
        print(f"  {col}: {n_out} outliera ({pct:.1f}%) | opseg: [{lower:.2f}, {upper:.2f}]")

    # Posebna napomena za pdays – vrednost 999 znaci "nije kontaktiran"
    pdays_999 = (dataset['pdays'] == 999).sum()
    print(f"\n  NAPOMENA – pdays=999: {pdays_999} redova ({pdays_999/len(dataset)*100:.1f}%)")
    print("  Vrednost 999 nije pravi outlier – oznacava da klijent nije prethodno kontaktiran.")


def check_correlations(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KORELACIJE IZMEDJU NUMERICKIH ATRIBUTA")
    print("=" * 60)

    corr = dataset[num_cols].corr()

    # Ispisujemo jake korelacije (|r| > 0.7), iskljucujuci dijagonalu
    print("\nJake korelacije (|r| > 0.7):")
    found = False
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {num_cols[i]} <-> {num_cols[j]}: r = {r:.3f}")
                found = True
    if not found:
        print("  Nema jakih korelacija.")

    # Korelacija numeričkih atributa sa ciljnom promenljivom
    print("\nKorelacija numeričkih atributa sa ciljnom promenljivom (y):")
    dataset_temp = dataset.copy()
    dataset_temp['y_bin'] = (dataset_temp['y'] == 'yes').astype(int)
    for col in num_cols:
        r, p = stats.pointbiserialr(dataset_temp[col], dataset_temp['y_bin'])
        print(f"  {col}: r = {r:.3f} (p = {p:.4f})")

    # Heatmap korelacione matrice
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Korelaciona matrica – numericki atributi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nSacuvano: correlation_heatmap.png")


def plot_target_distribution(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DISTRIBUCIJA CILJNE PROMENLJIVE")
    print("=" * 60)

    counts = dataset['y'].value_counts()
    pct = dataset['y'].value_counts(normalize=True) * 100
    print(counts)
    print(f"\nno:  {pct['no']:.1f}%")
    print(f"yes: {pct['yes']:.1f}%")
    print("\nNAPOMENA: Dataset je neuravnotezen – 'no' dominira.")
    print("Ovo znaci da model moze biti pristrasan ka predikciji 'no'.")
    print("Resenje: koristicemo SMOTE pri treniranju modela.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=['#2563EB', '#F59E0B'],
                edgecolor='white', linewidth=1.5)
    for i, (val, cnt) in enumerate(counts.items()):
        axes[0].text(i, cnt + 200, f'{cnt:,}', ha='center',
                     fontweight='bold', fontsize=10)
    axes[0].set_title('Broj uzoraka po klasi')
    axes[0].set_xlabel('Pretplata (y)')
    axes[0].set_ylabel('Broj klijenata')

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=['#2563EB', '#F59E0B'], startangle=90)
    axes[1].set_title('Procentualna raspodela klasa')

    plt.suptitle('Distribucija ciljne promenljive (y)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()
    print("Sacuvano: target_distribution.png")


def plot_numeric_distributions(dataset: pd.DataFrame):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for val, color in zip(['no', 'yes'], ['#2563EB', '#F59E0B']):
            axes[i].hist(dataset[dataset['y'] == val][col], bins=30,
                         alpha=0.7, color=color, label=val,
                         edgecolor='white', density=True)
        axes[i].set_title(col, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.suptitle('Distribucija numerickih atributa po ciljnoj promenljivoj',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=150)
    plt.close()
    print("Sacuvano: numeric_distributions.png")


def plot_categorical_vs_target(dataset: pd.DataFrame):
    # Za svaki kategoricki atribut prikazujemo % 'yes' po kategoriji
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        pct_yes = (dataset.groupby(col)['y']
                   .apply(lambda x: (x == 'yes').sum() / len(x) * 100)
                   .sort_values(ascending=False))

        axes[i].bar(range(len(pct_yes)), pct_yes.values,
                    color='#F59E0B', edgecolor='white')
        axes[i].set_xticks(range(len(pct_yes)))
        axes[i].set_xticklabels(pct_yes.index, rotation=45, ha='right', fontsize=7)
        axes[i].set_title(f'{col}', fontweight='bold')
        axes[i].set_ylabel('% yes')
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('Procenat pretplate (yes) po kategorickim atributima',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=150)
    plt.close()
    print("Sacuvano: categorical_vs_target.png")


def plot_boxplots(dataset: pd.DataFrame):
    # Boxplotovi pokazuju distribuciju i outlijere za svaki numericki atribut
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        data_no = dataset[dataset['y'] == 'no'][col]
        data_yes = dataset[dataset['y'] == 'yes'][col]

        axes[i].boxplot([data_no, data_yes], tick_labels=['no', 'yes'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#EFF6FF'),
                        medianprops=dict(color='#2563EB', linewidth=2))
        axes[i].set_title(col, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('Boxplot numerickih atributa po ciljnoj promenljivoj',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots.png", dpi=150)
    plt.close()
    print("Sacuvano: boxplots.png")


def crosstab_analysis(dataset: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CROSSTAB ANALIZA – KATEGORICKI ATRIBUTI vs CILJNA PROMENLJIVA")
    print("=" * 60)

    for col in cat_cols:
        print(f"\nAtribut: {col}")
        ct = pd.crosstab(dataset[col], dataset['y'], normalize='index').round(3) * 100
        ct.columns = ['% no', '% yes']
        print(ct.sort_values('% yes', ascending=False).to_string())


if __name__ == "__main__":
    dataset = load_data()
    basic_info(dataset)
    check_missing_and_unknown(dataset)
    check_outliers(dataset)
    check_correlations(dataset)
    plot_target_distribution(dataset)
    plot_numeric_distributions(dataset)
    plot_categorical_vs_target(dataset)
    plot_boxplots(dataset)
    crosstab_analysis(dataset)

    print("\n" + "=" * 60)
    print("EDA ZAVRSENA")
    print(f"Grafici sacuvani u: {FIGURES_DIR}")
    print("=" * 60)