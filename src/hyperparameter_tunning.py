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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_preparation import DATA_PATH, MODEL_DIR, RESULTS_DIR, FIGURES_DIR, get_data_and_preprocessor

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Isti kriterijum kao u model_training.py - F2 za yes klasu. Vidi obrazlozenje tamo.
SELECTION_METRIC = "val_f2_yes"
MIN_ACCEPTABLE_PRECISION = 0.20

# scoring za GridSearchCV: F2 favorizuje Recall (beta=2), sto odgovara cilju
# projekta (sto manje propustenih klijenata koji bi se pretplatili).
F2_SCORER = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)


def get_param_grids():
    return {
        # Logistic Regression - brz
        "Logistic Regression": {
            "classifier": LogisticRegression(
                max_iter=2000, solver="liblinear", random_state=42,
                class_weight="balanced"
            ),
            "params": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
            },
        },
        # Decision Tree - brz
        "Decision Tree": {
            "classifier": DecisionTreeClassifier(
                random_state=42, class_weight="balanced"
            ),
            "params": {
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__criterion": ["gini", "entropy"],
            },
        },
        # Random Forest - srednje brz
        # NAPOMENA: n_jobs=-1 je ranije bio postavljen i ovde i u
        # GridSearchCV-u ispod, sto izaziva ugnjezdeni paralelizam (nested
        # parallelism) - svaki GridSearchCV worker bi sam pokusao da
        # pokrene RandomForest na svim jezgrima, sto preopterecuje CPU i
        # moze biti SPORIJE nego sa jednim nivoom paralelizma. Zato je
        # ovde n_jobs=1 (paralelizam se desava samo na nivou GridSearchCV-a).
        "Random Forest": {
            "classifier": RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=1
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                # max_depth=5 dodat nakon eksperimenta - na ovom datasetu
                # plice stablo (manje sklono overfitting-u) daje bolji F2
                # nego dublje varijante (10, 16, None).
                "classifier__max_depth": [5, 10, 16, None],
                "classifier__max_features": ["sqrt"],
                "classifier__min_samples_split": [2, 10],
            },
        },
        # Gradient Boosting - skraceno (bio je najsporiji)
        "Gradient Boosting": {
            "classifier": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [0.8, 1.0],
            },
        },
    }


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
        "pr_auc": average_precision_score(y, y_prob),
        "classification_report": classification_report(y, y_pred, target_names=["no", "yes"], zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def get_data_splits():
    """Učitava ili kreira podeljene podatke"""
    splits_path = MODEL_DIR / "data_splits.joblib"

    if splits_path.exists():
        print("Učitavanje postojećih podeljenih podataka...")
        return joblib.load(splits_path)
    else:
        print("data_splits.joblib nije pronađen. Kreiranje novih podeljenih podataka...")
        X, y, preprocessor, _, _ = get_data_and_preprocessor(DATA_PATH)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
        splits = (X_train, X_val, X_test, y_train, y_val, y_test)
        joblib.dump(splits, splits_path)
        print(f"Podaci sačuvani u {splits_path}")
        return splits


def tune_models(X_train, y_train, X_val, y_val, preprocessor):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_models = {}
    rows = []

    for model_name, config in get_param_grids().items():
        print("\n" + "=" * 60)
        print("Podesavanje:", model_name)
        print("=" * 60)

        total_combinations = 1
        for param_values in config["params"].values():
            total_combinations *= len(param_values)
        print(f"Broj kombinacija za GridSearch: {total_combinations}")
        print(f"Očekivano trajanje: ~{total_combinations * 3} sekundi po modelu")
        print("-" * 40)

        model = build_pipeline(preprocessor, config["classifier"])

        grid_search = GridSearchCV(
            model,
            config["params"],
            cv=cv,
            scoring=F2_SCORER,
            n_jobs=-1,
            refit=True,
            verbose=2
        )

        print("Počinje GridSearch...")
        grid_search.fit(X_train, y_train)
        print("GridSearch završen!")

        tuned_model = grid_search.best_estimator_
        tuned_models[model_name] = tuned_model
        metrics = evaluate_model(tuned_model, X_val, y_val)

        print("\n" + "-" * 40)
        print("REZULTATI:")
        print("-" * 40)
        print("Najbolji CV F2 (yes):", round(grid_search.best_score_, 4))
        print("Najbolji parametri:", grid_search.best_params_)
        print("Validation F2 (yes):", round(metrics["f2_yes"], 4))
        print("Validation F1 (yes):", round(metrics["f1_yes"], 4))
        print("Validation Recall:", round(metrics["recall"], 4))
        print("Validation Precision:", round(metrics["precision"], 4))
        print("Validation Accuracy:", round(metrics["accuracy"], 4))
        print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))
        print("Validation PR-AUC:", round(metrics["pr_auc"], 4))
        if metrics["precision"] < MIN_ACCEPTABLE_PRECISION:
            print(f"⚠️  UPOZORENJE: Precision ({metrics['precision']:.4f}) je ispod "
                  f"prihvatljivog minimuma ({MIN_ACCEPTABLE_PRECISION}).")
        print("\nClassification report")
        print(metrics["classification_report"])
        print("\nConfusion matrix")
        print(metrics["confusion_matrix"])

        rows.append({
            "model": model_name,
            "best_cv_f2_yes": grid_search.best_score_,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1_yes": metrics["f1_yes"],
            "val_f2_yes": metrics["f2_yes"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "best_params": str(grid_search.best_params_),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "tuned_validation_results.csv", index=False)
    return tuned_models, results_df


def save_tuned_models(tuned_models, results_df):
    for model_name, model in tuned_models.items():
        fname = model_name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"tuned_{fname}.joblib")

    best_row = results_df.sort_values(by=SELECTION_METRIC, ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = tuned_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_tuned_model.joblib")
    joblib.dump(best_model, MODEL_DIR / "final_model.joblib")
    (MODEL_DIR / "best_tuned_model_name.txt").write_text(best_name, encoding="utf-8")

    print("\n" + "=" * 60)
    print("NAJBOLJI MODEL NAKON TUNINGA")
    print("=" * 60)
    print("Najbolji podeseni model:", best_name)
    print("Validation F2 (yes):", round(best_row["val_f2_yes"], 4))
    print("Validation F1 (yes):", round(best_row["val_f1_yes"], 4))
    print("Validation Recall:", round(best_row["val_recall"], 4))
    print("Validation Precision:", round(best_row["val_precision"], 4))
    if best_row["val_precision"] < MIN_ACCEPTABLE_PRECISION:
        print(f"⚠️  UPOZORENJE: izabrani model ima Precision ispod {MIN_ACCEPTABLE_PRECISION}.")
    print("Sacuvan kao models/best_tuned_model.joblib i models/final_model.joblib")


def plot_tuning_results(results_df):
    """Bar plot sa svim kljucnim metrikama (ne samo F2) za balansiran prikaz."""
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
    ax.set_title("Poredjenje podesenih modela - sve metrike (rangirano po F2)")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tuned_model_comparison.png", dpi=150)
    plt.close()


def compare_with_untuned(results_df):
    """Poređenje sa netjuniranim modelima iz model_training.py"""
    try:
        untuned_path = RESULTS_DIR / "all_features_validation_results.csv"
        if untuned_path.exists():
            untuned_df = pd.read_csv(untuned_path)

            comparison = pd.merge(
                untuned_df[["model", SELECTION_METRIC]],
                results_df[["model", SELECTION_METRIC]],
                on="model",
                suffixes=("_untuned", "_tuned")
            )
            comparison["improvement"] = (
                comparison[f"{SELECTION_METRIC}_tuned"] - comparison[f"{SELECTION_METRIC}_untuned"]
            )

            print("\n" + "=" * 60)
            print("POREDJENJE TUNED VS UNTUNED MODELA")
            print("=" * 60)
            print(comparison.to_string(index=False))

            comparison.to_csv(RESULTS_DIR / "tuning_improvement_comparison.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(comparison))
            width = 0.35

            ax.bar(x, comparison[f"{SELECTION_METRIC}_untuned"], width, label="Untuned", alpha=0.7)
            ax.bar([i + width for i in x], comparison[f"{SELECTION_METRIC}_tuned"], width, label="Tuned", alpha=0.7)

            ax.set_xlabel("Model")
            ax.set_ylabel("Validation F2 (yes)")
            ax.set_title("Poređenje performansi pre i posle tuning-a")
            ax.set_xticks([i + width / 2 for i in x])
            ax.set_xticklabels(comparison["model"], rotation=15)
            ax.legend()

            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "tuning_comparison.png", dpi=150)
            plt.close()

            return comparison
    except Exception as e:
        print(f"⚠️  Nije moguće uporediti sa untuned modelima: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("HIPERPARAMETAR TUNING")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits()

    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        print("Preprocessor učitano iz models/preprocessor.joblib")
    else:
        print("Preprocessor nije pronađen, kreiram novi...")
        X, y, preprocessor, _, _ = get_data_and_preprocessor(DATA_PATH)
        joblib.dump(preprocessor, preprocessor_path)

    tuned_models, results_df = tune_models(X_train, y_train, X_val, y_val, preprocessor)

    print("\n" + "=" * 60)
    print("POREDJENJE PODEŠENIH MODELA")
    print("=" * 60)
    print(results_df.sort_values(by=SELECTION_METRIC, ascending=False).to_string(index=False))

    save_tuned_models(tuned_models, results_df)
    plot_tuning_results(results_df)
    compare_with_untuned(results_df)

    print("\n" + "=" * 60)
    print("HIPERPARAMETAR TUNING ZAVRŠEN")
    print("=" * 60)
    print(f"Rezultati sačuvani u: {RESULTS_DIR}")
    print(f"Modeli sačuvani u: {MODEL_DIR}")