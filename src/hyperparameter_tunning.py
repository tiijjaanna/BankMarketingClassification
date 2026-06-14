from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_xgb_pipeline() -> ImbPipeline:
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )),
    ])


def run_grid_search(X_train, y_train) -> GridSearchCV:
    print("=" * 60)
    print("PODESAVANJE HIPERPARAMETARA – XGBoost")
    print("=" * 60)

    print("""
Hiperparametri koje podešavamo:

  n_estimators   – broj stabala u ansamblu
                   vise stabala = stabilniji model, ali sporiji trening
                   malo: moze biti underfitting
                   puno: dobro, ali od nekog broja nema poboljsanja

  max_depth      – maksimalna dubina svakog stabla
                   plitko (3): jednostavniji model, manje overfittinga
                   duboko (7): kompleksniji, moze zapamtiti suma

  learning_rate  – korak ucenja (eta)
                   manji: stabilnije ucenje, potrebno vise stabala
                   veci: brze, ali moze da promasuje minimum

  subsample      – procenat podataka za svako stablo
                   < 1.0: uvodi slucajnost, smanjuje overfitting

  scale_pos_weight – tezina manjinske klase (yes=11%)
                   neg/pos = 34806/4598 ≈ 7.57
                   govori modelu da "yes" primeri vise vrede
    """)

    # Parametri koje pretražujemo
    # Prefiks "classifier__" govori Pipelineu koji korak menjamo
    #
    # scale_pos_weight fiksiramo na 7.57 jer je to matematički tačna vrednost
    # za naš dataset: neg/pos = 34806/4598 ≈ 7.57
    # Govori modelu da "yes" primeri vrede 7.57x više od "no"
    # Ne ima smisla pretražiti druge vrednosti jer je ova izvedena iz podataka.
    param_grid = {
        "classifier__n_estimators":     [100, 200],
        "classifier__max_depth":        [3, 5],
        "classifier__learning_rate":    [0.05, 0.1],
        "classifier__subsample":        [0.8, 1.0],
        "classifier__scale_pos_weight": [7.57],
    }

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Ukupno kombinacija: {total}")
    print(f"Sa 5-fold CV = {total * 5} treniranja")
    print("\nPokretanje GridSearchCV (ovo moze potrajati 5-15 minuta)...\n")

    pipeline = build_xgb_pipeline()

    # StratifiedKFold unutar GridSearchCV — evaluacija se vrsi
    # SAMO na trening skupu, validacioni i test ostaju netaknuti
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",   # biramo po Macro F1 jer nas zanima balans klasa
        n_jobs=-1,            # sva CPU jezgra
        verbose=1,
        refit=True            # automatski trenira best model na celom trening skupu
    )

    grid_search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("REZULTATI GRID SEARCHA")
    print("=" * 60)
    print(f"\nNajbolji CV Macro F1: {grid_search.best_score_:.4f}")
    print("\nNajbolji parametri:")
    for k, v in grid_search.best_params_.items():
        param = k.replace("classifier__", "")
        print(f"  {param}: {v}")

    return grid_search


def evaluate_on_validation(pipeline, X_val, y_val):
    print("\n" + "=" * 60)
    print("EVALUACIJA PODEŠENOG MODELA – VALIDACIONI SKUP")
    print("=" * 60)

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    acc   = accuracy_score(y_val, y_pred)
    prec  = precision_score(y_val, y_pred, zero_division=0)
    rec   = recall_score(y_val, y_pred)
    f1mac = f1_score(y_val, y_pred, average="macro")
    f1yes = f1_score(y_val, y_pred, pos_label=1)
    roc   = roc_auc_score(y_val, y_prob)

    print(f"\n  Accuracy:      {acc:.4f}")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")
    print(f"  Macro F1:      {f1mac:.4f}")
    print(f"  F1 (yes):      {f1yes:.4f}")
    print(f"  ROC-AUC:       {roc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                target_names=["no", "yes"],
                                zero_division=0))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"\n  Propustenih klijenata (FN): {fn} "
          f"({fn/(fn+tp)*100:.1f}% od svih 'yes')")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_macro": round(f1mac, 4),
        "f1_yes": round(f1yes, 4),
        "roc_auc": round(roc, 4),
    }


def compare_with_baseline(tuned_results: dict, trained_models: dict,
                           X_val, y_val):
    print("\n" + "=" * 60)
    print("POREDJENJE: PODESEN XGBoost vs. OSTALI MODELI")
    print("=" * 60)

    comparison = []

    # Dodajemo podešen XGBoost
    comparison.append({
        "model": "XGBoost (podesen)",
        **tuned_results
    })

    # Dodajemo ostale modele sa validacionog skupa
    for name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        comparison.append({
            "model": name,
            "accuracy":  round(accuracy_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_val, y_pred), 4),
            "f1_macro":  round(f1_score(y_val, y_pred, average="macro"), 4),
            "f1_yes":    round(f1_score(y_val, y_pred, pos_label=1), 4),
            "roc_auc":   round(roc_auc_score(y_val, y_prob), 4),
        })

    df = pd.DataFrame(comparison).sort_values("f1_macro", ascending=False)

    print(f"\n{'Model':<25} {'Macro F1':>10} {'F1 yes':>8} "
          f"{'Recall':>8} {'Precision':>10} {'ROC-AUC':>10}")
    print("-" * 75)
    for _, row in df.iterrows():
        marker = " ← POBEDNIK" if row["model"] == df.iloc[0]["model"] else ""
        print(f"{row['model']:<25} {row['f1_macro']:>10.4f} "
              f"{row['f1_yes']:>8.4f} {row['recall']:>8.4f} "
              f"{row['precision']:>10.4f} {row['roc_auc']:>10.4f}{marker}")

    return df


def plot_tuning_results(grid_search: GridSearchCV):
    # Prikazujemo kako se menjao CV F1 u zavisnosti od n_estimators i max_depth
    results_df = pd.DataFrame(grid_search.cv_results_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafik 1: n_estimators vs CV F1
    for depth in [3, 5, 7]:
        mask = results_df["param_classifier__max_depth"] == depth
        subset = results_df[mask].groupby(
            "param_classifier__n_estimators")["mean_test_score"].mean()
        axes[0].plot(subset.index, subset.values,
                     marker='o', label=f"max_depth={depth}")

    axes[0].set_title("n_estimators vs CV Macro F1", fontweight="bold")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("CV Macro F1")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Grafik 2: learning_rate vs CV F1
    for depth in [3, 5, 7]:
        mask = results_df["param_classifier__max_depth"] == depth
        subset = results_df[mask].groupby(
            "param_classifier__learning_rate")["mean_test_score"].mean()
        axes[1].plot(subset.index, subset.values,
                     marker='o', label=f"max_depth={depth}")

    axes[1].set_title("learning_rate vs CV Macro F1", fontweight="bold")
    axes[1].set_xlabel("learning_rate")
    axes[1].set_ylabel("CV Macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("GridSearch rezultati – XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tuning_results.png", dpi=150)
    plt.close()
    print("\nSacuvano: tuning_results.png")


def save_tuning_results(grid_search: GridSearchCV, comparison_df: pd.DataFrame):
    # Cuvamo top 10 kombinacija parametara
    results_df = pd.DataFrame(grid_search.cv_results_)
    cols = [c for c in results_df.columns
            if c.startswith("param_") or c in
            ["mean_test_score", "std_test_score", "rank_test_score"]]
    top10 = results_df[cols].sort_values(
        "rank_test_score").head(10)
    top10.columns = [c.replace("param_classifier__", "") for c in top10.columns]
    top10.to_csv(RESULTS_DIR / "tuning_top10.csv", index=False)

    comparison_df.to_csv(RESULTS_DIR / "tuning_comparison.csv", index=False)

    print("\nTop 10 kombinacija parametara:")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    print("Ucitavanje podataka...")
    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(
        MODEL_DIR / "data_splits.joblib"
    )
    trained_models = joblib.load(MODEL_DIR / "all_models.joblib")

    print(f"Trening skup:     {X_train.shape[0]} uzoraka")
    print(f"Validacioni skup: {X_val.shape[0]} uzoraka")
    print(f"Test skup:        {X_test.shape[0]} uzoraka")

    # Grid Search na trening skupu
    grid_search = run_grid_search(X_train, y_train)

    # Evaluacija na validacionom skupu
    best_pipeline = grid_search.best_estimator_
    tuned_results = evaluate_on_validation(best_pipeline, X_val, y_val)

    # Poredjenje sa ostalim modelima
    comparison_df = compare_with_baseline(
        tuned_results, trained_models, X_val, y_val
    )

    # Grafici i cuvanje rezultata
    plot_tuning_results(grid_search)
    save_tuning_results(grid_search, comparison_df)

    # Cuvamo podesen model
    joblib.dump(best_pipeline, MODEL_DIR / "tuned_xgboost.joblib")
    joblib.dump(
        (X_train, X_val, X_test, y_train, y_val, y_test),
        MODEL_DIR / "data_splits.joblib"
    )

    print("\n" + "=" * 60)
    print("PODESAVANJE ZAVRSENO")
    print(f"Podesen model sacuvan kao: models/tuned_xgboost.joblib")
    print("Kada si zadovoljna rezultatima, pokreni model_evaluation.py")
    print("=" * 60)