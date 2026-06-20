from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score,
    make_scorer, precision_score, recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_preparation import DATA_PATH, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, ECONOMIC_FEATURES, get_data_and_preprocessor

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Kriterijum po kome se bira "najbolji" model: F2 za yes klasu (val_f2_yes).
#
# F_beta = (1+beta^2) * (Precision*Recall) / (beta^2*Precision + Recall)
# Za beta=2: F2 = 5 * (P*R) / (4P + R)  -> Recall ima 4x vecu tezinu od
# Precision-a u formuli. Ovo direktno odrazava temu projekta (predikcija
# koliko ce banka klijenata "upecati"): propusten klijent koji bi se
# pretplatio (FN) smatra se skupljom greskom nego suvisan poziv (FP), pa F2
# nagradjuje modele/parametre koji bolje hvataju pozitivnu klasu, bez da
# potpuno zanemaruje Precision (kao sto bi cist Recall kriterijum mogao).
#
# SANITY-CHECK: kriterijum izbora ostaje F2, ali se uz njega prati i
# Precision - ako bi neki kandidat imao ekstremno nizak Precision (npr.
# model koji gotovo uvek predvidja "yes"), to bi bilo vidljivo u
# MIN_ACCEPTABLE_PRECISION proveri ispod, kako se ne bi nesvesno izabrao
# poslovno beskoristan model. Na trenutnom skupu od 4 modela ovo se ne
# aktivira (svi imaju Precision > 0.30), ali ostaje kao zastita za buduce
# izmene hiperparametara.
SELECTION_METRIC = "val_f2_yes"
MIN_ACCEPTABLE_PRECISION = 0.20

# scoring za cross_val_score: F2 favorizuje Recall (beta=2), isti kriterijum
# kao SELECTION_METRIC.
F2_SCORER = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)


def get_models():
    return {
        # class_weight="balanced" kompenzuje neuravnotezenost klasa (88%/12%)
        # zajedno sa SMOTE-om u pipeline-u - SMOTE balansira podatke,
        # class_weight dodatno balansira gresku modela.
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="liblinear", random_state=42,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced"
        ),
        # GradientBoostingClassifier u sklearn-u NE PODRZAVA class_weight
        # parametar, pa se za njega oslanjamo iskljucivo na SMOTE.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    print("\nDimenzije skupova")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(preprocessor, classifier):
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1_yes": f1_score(y, y_pred, pos_label=1),
        "f2_yes": fbeta_score(y, y_pred, beta=2, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        # PR-AUC (average precision) je informativniji od ROC-AUC kada je
        # pozitivna klasa retka (~11%) - baseline za PR-AUC je udeo
        # pozitivne klase, za razliku od ROC-AUC ciji je baseline uvek 0.5.
        "pr_auc": average_precision_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred, target_names=["no", "yes"], zero_division=0),
    }


def train_models(X_train, X_val, y_train, y_val, preprocessor, experiment_name):
    trained_models = {}
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, classifier in get_models().items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)
        model = build_pipeline(preprocessor, classifier)

        # CV scoring prati isti kriterijum kao i finalni izbor modela (F2 za
        # yes klasu).
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=F2_SCORER, n_jobs=-1)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        metrics = evaluate_model(model, X_val, y_val)

        print("CV F2 (yes):", round(cv_scores.mean(), 4))
        print("Validation Accuracy:", round(metrics["accuracy"], 4))
        print("Validation Precision:", round(metrics["precision"], 4))
        print("Validation Recall:", round(metrics["recall"], 4))
        print("Validation F1 (yes):", round(metrics["f1_yes"], 4))
        print("Validation F2 (yes):", round(metrics["f2_yes"], 4))
        print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))
        print("Validation PR-AUC:", round(metrics["pr_auc"], 4))
        if metrics["precision"] < MIN_ACCEPTABLE_PRECISION:
            print(f"⚠️  UPOZORENJE: Precision ({metrics['precision']:.4f}) je ispod "
                  f"prihvatljivog minimuma ({MIN_ACCEPTABLE_PRECISION}). Model moze biti "
                  f"poslovno beskoristan uprkos visokom F2/Recall-u.")
        print("\nClassification report")
        print(metrics["classification_report"])
        print("\nConfusion matrix")
        print(metrics["confusion_matrix"])

        rows.append({
            "experiment": experiment_name,
            "model": model_name,
            "cv_f2_yes": cv_scores.mean(),
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1_yes": metrics["f1_yes"],
            "val_f2_yes": metrics["f2_yes"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / f"{experiment_name}_validation_results.csv", index=False)
    print(f"\nPoredjenje modela (rangirano po {SELECTION_METRIC})")
    print(results_df.sort_values(by=SELECTION_METRIC, ascending=False))
    return trained_models, results_df


def plot_results(results_df, filename):
    """Bar plot koji prikazuje SVE kljucne metrike jedna pored druge (ne
    samo F2), da bi se izbor modela mogao 'izbalansirano' sagledati."""
    sorted_df = results_df.sort_values(by=SELECTION_METRIC, ascending=False)
    metrics_to_plot = ["val_accuracy", "val_precision", "val_recall", "val_f1_yes", "val_f2_yes"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "F2"]

    x = range(len(sorted_df))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        offset = (i - len(metrics_to_plot) / 2) * width
        ax.bar([p + offset for p in x], sorted_df[metric], width, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sorted_df["model"], rotation=15)
    ax.set_ylabel("Vrednost metrike")
    ax.set_title("Poredjenje modela - sve metrike (rangirano po F2)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def save_best_model(trained_models, results_df, suffix=""):
    # Biramo model po F2 (yes) - vidi obrazlozenje uz SELECTION_METRIC gore.
    best_row = results_df.sort_values(by=SELECTION_METRIC, ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = trained_models[best_name]
    model_path = MODEL_DIR / (f"best_model_{suffix}.joblib" if suffix else "best_model.joblib")
    name_path = MODEL_DIR / (f"best_model_name_{suffix}.txt" if suffix else "best_model_name.txt")
    joblib.dump(best_model, model_path)
    name_path.write_text(best_name, encoding="utf-8")

    print("\nNajbolji model:", best_name)
    print(f"Validation F2 (yes): {best_row['val_f2_yes']:.4f}")
    print(f"Validation F1 (yes): {best_row['val_f1_yes']:.4f}")
    print(f"Validation Recall: {best_row['val_recall']:.4f}")
    print(f"Validation Precision: {best_row['val_precision']:.4f}")
    print(f"Validation Accuracy: {best_row['val_accuracy']:.4f}")
    if best_row["val_precision"] < MIN_ACCEPTABLE_PRECISION:
        print(f"⚠️  UPOZORENJE: izabrani model ima Precision ispod {MIN_ACCEPTABLE_PRECISION} - "
              f"proveriti da li je izbor poslovno opravdan pre finalne upotrebe.")
    print("Sacuvan kao:", model_path)
    return best_name, best_model


def run_experiment(experiment_name, selected_features=None):
    print("\n" + "#" * 70)
    print("EKSPERIMENT:", experiment_name)
    print("#" * 70)
    X, y, preprocessor, categorical_features, numeric_features = get_data_and_preprocessor(DATA_PATH, selected_features=selected_features)
    print("\nKategorijski atributi:")
    print(categorical_features)
    print("\nNumericki atributi:")
    print(numeric_features)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    trained_models, results_df = train_models(X_train, X_val, y_train, y_val, preprocessor, experiment_name)
    plot_results(results_df, f"{experiment_name}_model_comparison.png")
    save_best_model(trained_models, results_df, suffix=experiment_name)
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "trained_models": trained_models, "results_df": results_df,
    }


def get_trained_models():
    """Vraća sve istrenirane modele ako postoje"""
    models_path = MODEL_DIR / "all_models.joblib"
    if models_path.exists():
        return joblib.load(models_path)
    return None


def get_best_model():
    """Vraća najbolji model"""
    for name in ["best_tuned_model.joblib", "best_model.joblib", "final_model.joblib"]:
        path = MODEL_DIR / name
        if path.exists():
            return joblib.load(path)
    return None


if __name__ == "__main__":
    # Glavni eksperiment: svi atributi osim duration (i redundantnih
    # ekonomskih atributa, uklonjenih unutar get_data_and_preprocessor).
    all_exp = run_experiment("all_features")

    # Cuvamo splitove iz glavnog eksperimenta za tuning/evaluation.
    joblib.dump((all_exp["X_train"], all_exp["X_val"], all_exp["X_test"], all_exp["y_train"], all_exp["y_val"], all_exp["y_test"]), MODEL_DIR / "data_splits.joblib")
    joblib.dump(all_exp["trained_models"], MODEL_DIR / "all_models.joblib")

    # Dodatni eksperiment: bez preostalih ekonomskih atributa (i euribor3m),
    # da se kvantitativno potvrdi koliko ekonomska grupa doprinosi modelu.
    all_features = list(all_exp["X_train"].columns)
    no_economic_features = [c for c in all_features if c not in ECONOMIC_FEATURES]
    no_econ_exp = run_experiment("without_economic_features", selected_features=no_economic_features)

    comparison = pd.concat([all_exp["results_df"], no_econ_exp["results_df"]], ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "feature_group_experiment_comparison.csv", index=False)
    print("\nPOREDJENJE EKSPERIMENATA: sa euribor3m vs bez svih ekonomskih atributa")
    print(comparison.sort_values(by=["model", "experiment"]))

    # Kao glavni model za nastavak cuvamo najbolji iz all_features eksperimenata.
    save_best_model(all_exp["trained_models"], all_exp["results_df"], suffix="")