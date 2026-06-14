from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data_preparation import get_prepared_data

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_models() -> dict:
    return {
        # Logistic Regression – baseline model, brz i interpretabilan.
        # class_weight="balanced" kompenzuje neuravnotezenost klasa (88%/12%)
        # jer SMOTE primenjujemo samo na trening, LR i dalje moze imati koristi
        # od eksplicitnog balansiranja tezina klasa
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),

        # Decision Tree – interpretabilan, lako vizualizabilan.
        # max_depth=10 sprecava overfitting (bez ogranicenja stablo moze
        # da zapamti svaki primer iz trening skupa)
        # class_weight="balanced" – isto kao kod LR
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            class_weight="balanced"
        ),

        # Random Forest – kombinuje vise stabala (n_estimators=100).
        # Svako stablo se trenira na random poduzorku podataka i atributa.
        # Otporan na overfitting, daje feature importance.
        # n_jobs=-1 – koristi sva dostupna CPU jezgra za brzinu
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        # XGBoost – trenutno jedan od najjacih algoritama za tabelarne podatke.
        # scale_pos_weight kompenzuje neuravnotezenost:
        # neg/pos = 34806/4598 ≈ 7.57 → daje veci znacaj manjinskoj klasi
        # eval_metric="logloss" – metrika za pracenje tokom treninga
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            scale_pos_weight=34806 / 4598,
            eval_metric="logloss",
            verbosity=0
        ),
    }


def split_data(X, y):
    # Delimo na 3 skupa: 70% trening, 15% validacioni, 15% test
    # stratify=y cuva proporciju klasa (88%/12%) u svakom skupu
    #
    # Zasto 3 skupa?
    # - trening: model uci na ovim podacima
    # - validacioni: biramo koji model je najbolji (bez "trosenja" test skupa)
    # - test: koristimo JEDNOM na kraju za finalnu ocenu
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Trening skup:     {X_train.shape[0]} uzoraka")
    print(f"Validacioni skup: {X_val.shape[0]} uzoraka")
    print(f"Test skup:        {X_test.shape[0]} uzoraka")
    print(f"\nRaspodela y_train:\n{y_train.value_counts()}")
    print(f"\nRaspodela y_val:\n{y_val.value_counts()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(classifier) -> ImbPipeline:
    # Pipeline redosled:
    # 1. StandardScaler – normalizuje sve numericke vrednosti na isti opseg
    #    (bitno za Logistic Regression i Distance-based modele)
    # 2. SMOTE – sinteticki generise uzorke manjinske klase (yes=11%)
    #    SMOTE se primenjuje SAMO na trening podacima
    #    (zato koristimo ImbPipeline iz imbalanced-learn, a ne sklearn Pipeline)
    # 3. Classifier – trenira model na balansiranim podacima
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])


def train_and_validate(X_train, X_val, y_train, y_val) -> tuple[dict, dict]:
    models = get_models()
    trained_pipelines = {}
    val_results = {}

    for name, classifier in models.items():
        print("\n" + "=" * 55)
        print(f"  {name}")
        print("=" * 55)

        pipeline = build_pipeline(classifier)

        # Cross-validacija na trening skupu (5 foldova)
        # Daje stabilniju procenu od jednog trening/val split-a
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro"
        )
        print(f"  CV Macro F1 (trening): {cv_scores.mean():.4f} "
              f"(+/- {cv_scores.std():.4f})")

        # Treniranje na celom trening skupu
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluacija na VALIDACIONOM skupu
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        acc      = accuracy_score(y_val, y_pred)
        prec     = precision_score(y_val, y_pred, zero_division=0)
        rec      = recall_score(y_val, y_pred)
        f1_mac   = f1_score(y_val, y_pred, average="macro")
        f1_yes   = f1_score(y_val, y_pred, pos_label=1)
        roc      = roc_auc_score(y_val, y_prob)

        print(f"\n  Rezultati na VALIDACIONOM skupu:")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Precision:     {prec:.4f}")
        print(f"  Recall:        {rec:.4f}")
        print(f"  Macro F1:      {f1_mac:.4f}")
        print(f"  F1 (yes):      {f1_yes:.4f}")
        print(f"  ROC-AUC:       {roc:.4f}")

        print(f"\n  Classification Report (validacioni skup):")
        print(classification_report(y_val, y_pred,
                                    target_names=["no", "yes"],
                                    zero_division=0))

        print(f"  Confusion Matrix (validacioni skup):")
        print(confusion_matrix(y_val, y_pred))

        val_results[name] = {
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1_macro":  round(f1_mac, 4),
            "f1_yes":    round(f1_yes, 4),
            "roc_auc":   round(roc, 4),
            "cv_f1_macro": round(cv_scores.mean(), 4),
        }

    return trained_pipelines, val_results


def print_summary(val_results: dict):
    print("\n" + "=" * 55)
    print("REZIME – VALIDACIONI SKUP (sortirano po Macro F1)")
    print("=" * 55)
    print(f"{'Model':<22} {'CV F1':>8} {'Val F1':>8} {'F1 yes':>8} {'ROC':>8} {'Recall':>8}")
    print("-" * 55)

    sorted_results = sorted(val_results.items(),
                            key=lambda x: x[1]["f1_macro"], reverse=True)
    for i, (name, r) in enumerate(sorted_results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "   "
        print(f"{medal} {name:<19} {r['cv_f1_macro']:>8.4f} {r['f1_macro']:>8.4f} "
              f"{r['f1_yes']:>8.4f} {r['roc_auc']:>8.4f} {r['recall']:>8.4f}")

    print("\nNajbolji model prema Macro F1:", sorted_results[0][0])


def plot_comparison(val_results: dict):
    df = pd.DataFrame(val_results).T.reset_index()
    df.columns = ["model"] + list(df.columns[1:])

    metrics = ["cv_f1_macro", "f1_macro", "f1_yes", "roc_auc"]
    labels  = ["CV Macro F1", "Val Macro F1", "Val F1 (yes)", "ROC-AUC"]
    colors  = ["#2563EB", "#F59E0B", "#10B981", "#EF4444"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = df[metric].astype(float)
        bars = axes[i].bar(df["model"], vals, color=color,
                           edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
        axes[i].set_title(label, fontweight="bold")
        axes[i].set_ylim(0, 1.1)
        axes[i].set_xticklabels(df["model"], rotation=15, ha="right")
        axes[i].grid(axis="y", alpha=0.3)

    plt.suptitle("Poredjenje modela – Validacioni skup",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_val.png", dpi=150)
    plt.close()
    print("\nSacuvano: model_comparison_val.png")


def save_models(trained_pipelines: dict, val_results: dict):
    # Cuvamo sve modele
    for name, pipeline in trained_pipelines.items():
        fname = name.lower().replace(" ", "_")
        path = MODEL_DIR / f"{fname}.joblib"
        joblib.dump(pipeline, path)

    # Cuvamo najbolji model prema validacionom F1
    best_name = max(val_results, key=lambda x: val_results[x]["f1_macro"])
    best_pipeline = trained_pipelines[best_name]
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")

    print(f"\nNajbolji model: {best_name}")
    print(f"Sacuvan kao: models/best_model.joblib")
    return best_name


if __name__ == "__main__":
    print("Ucitavanje i priprema podataka...")
    X, y = get_prepared_data(DATA_PATH)
    print(f"Oblik X: {X.shape} | Raspodela y: {dict(y.value_counts())}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\n" + "=" * 55)
    print("TRENIRANJE MODELA")
    print("=" * 55)

    trained_pipelines, val_results = train_and_validate(
        X_train, X_val, y_train, y_val
    )

    print_summary(val_results)
    plot_comparison(val_results)

    best_name = save_models(trained_pipelines, val_results)

    # Cuvamo sve podatke za evaluaciju i hyperparameter tuning
    joblib.dump(
        (X_train, X_val, X_test, y_train, y_val, y_test),
        MODEL_DIR / "data_splits.joblib"
    )
    joblib.dump(trained_pipelines, MODEL_DIR / "all_models.joblib")

    results_df = pd.DataFrame(val_results).T
    results_df.to_csv(RESULTS_DIR / "validation_results.csv")

    print("\n" + "=" * 55)
    print("TRENIRANJE ZAVRSENO")
    print(f"Kada si zadovoljna rezultatima, pokreni model_evaluation.py")
    print("=" * 55)