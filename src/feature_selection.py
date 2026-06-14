from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Koliko top atributa zadrzavamo u reduktivnom modelu
TOP_N = 10


def get_feature_importance(model: ImbPipeline, feature_names: list) -> pd.DataFrame:
    # XGBoost cuva feature importance u named_steps["classifier"]
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, title: str, filename: str):
    top = importance_df.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"][::-1], top["importance"][::-1],
             color="#2563EB", edgecolor="white", linewidth=1.2)
    plt.xlabel("Feature Importance (Gain)", fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()
    print(f"Sacuvano: {filename}")


def train_reduced_model(X_train, X_test, y_train, y_test,
                        top_features: list) -> dict:
    print(f"\n{'='*60}")
    print(f"TRENIRANJE MODELA SA TOP {TOP_N} ATRIBUTA")
    print(f"{'='*60}")
    print(f"Odabrani atributi: {top_features}\n")

    X_train_red = X_train[top_features]
    X_test_red = X_test[top_features]

    # Koristimo iste hiperparametre kao podesen model
    try:
        tuned = joblib.load(MODEL_DIR / "tuned_model.joblib")
        xgb_params = tuned.named_steps["classifier"].get_params()
    except FileNotFoundError:
        xgb_params = {"n_estimators": 100, "max_depth": 5,
                      "learning_rate": 0.1, "subsample": 0.8}

    pipeline_red = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(
            **{k: v for k, v in xgb_params.items()
               if k in ["n_estimators", "max_depth", "learning_rate", "subsample"]},
            random_state=42, eval_metric="logloss", verbosity=0
        )),
    ])

    pipeline_red.fit(X_train_red, y_train)
    y_pred_red = pipeline_red.predict(X_test_red)

    results = {
        "accuracy": accuracy_score(y_test, y_pred_red),
        "f1_macro": f1_score(y_test, y_pred_red, average="macro"),
        "f1_yes": f1_score(y_test, y_pred_red, pos_label=1),
    }

    print("Rezultati modela sa redukovanim atributima:")
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    print(f"  Macro F1:   {results['f1_macro']:.4f}")
    print(f"  F1 (yes):   {results['f1_yes']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_red,
                                target_names=["no", "yes"], zero_division=0))

    joblib.dump(pipeline_red, MODEL_DIR / "reduced_model.joblib")
    return results


def compare_full_vs_reduced(full_results: dict, reduced_results: dict,
                             feature_names: list):
    print(f"\n{'='*60}")
    print("POREDJENJE: SVI ATRIBUTI vs TOP ATRIBUTI")
    print(f"{'='*60}")

    comparison = pd.DataFrame({
        "Metrika": ["Accuracy", "Macro F1", "F1 (yes)"],
        f"Svi atributi ({len(feature_names)})": [
            full_results["accuracy"],
            full_results["f1_macro"],
            full_results["f1_yes"],
        ],
        f"Top {TOP_N} atributa": [
            reduced_results["accuracy"],
            reduced_results["f1_macro"],
            reduced_results["f1_yes"],
        ],
    })

    print(comparison.to_string(index=False))

    # Grafik poredjenja
    metrics = ["Accuracy", "Macro F1", "F1 (yes)"]
    full_vals = [full_results["accuracy"], full_results["f1_macro"], full_results["f1_yes"]]
    red_vals = [reduced_results["accuracy"], reduced_results["f1_macro"], reduced_results["f1_yes"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, full_vals, width, label=f"Svi atributi ({len(feature_names)})",
                   color="#2563EB", edgecolor="white")
    bars2 = ax.bar(x + width/2, red_vals, width, label=f"Top {TOP_N} atributa",
                   color="#F59E0B", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("Poredjenje: svi vs. najbitniji atributi", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "full_vs_reduced.png", dpi=150)
    plt.close()
    print("\nSacuvano: full_vs_reduced.png")

    comparison.to_csv(RESULTS_DIR / "full_vs_reduced.csv", index=False)


if __name__ == "__main__":
    print("Ucitavanje podataka...")
    X, y = get_prepared_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ucitavamo podesen model (ili best_model ako tuning nije pokrenut)
    try:
        best_model = joblib.load(MODEL_DIR / "tuned_model.joblib")
        print("Ucitan podesen model (tuned_model.joblib)")
    except FileNotFoundError:
        best_model = joblib.load(MODEL_DIR / "best_model.joblib")
        print("Ucitan najbolji model (best_model.joblib)")

    # Feature importance na svim atributima
    feature_names = list(X.columns)
    importance_df = get_feature_importance(best_model, feature_names)

    print("\nTop 10 najbitnijih atributa:")
    print(importance_df.head(10).to_string(index=False))

    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df,
                            "Feature Importance – XGBoost (svi atributi)",
                            "feature_importance_full.png")

    # Rezultati modela sa svim atributima
    y_pred_full = best_model.predict(X_test)
    full_results = {
        "accuracy": accuracy_score(y_test, y_pred_full),
        "f1_macro": f1_score(y_test, y_pred_full, average="macro"),
        "f1_yes": f1_score(y_test, y_pred_full, pos_label=1),
    }

    # Treniranje sa top N atributima
    top_features = importance_df.head(TOP_N)["feature"].tolist()
    reduced_results = train_reduced_model(
        X_train, X_test, y_train, y_test, top_features
    )

    # Feature importance redukovanog modela
    reduced_model = joblib.load(MODEL_DIR / "reduced_model.joblib")
    importance_red = get_feature_importance(reduced_model, top_features)
    plot_feature_importance(importance_red,
                            f"Feature Importance – XGBoost (top {TOP_N} atributa)",
                            "feature_importance_reduced.png")

    # Poredjenje
    compare_full_vs_reduced(full_results, reduced_results, feature_names)

    print("\nOdabir atributa zavrsen!")