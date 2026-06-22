import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import OneHotEncoder

from data_preparation import (
    DATA_PATH, MODEL_DIR, RESULTS_DIR, FIGURES_DIR,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_data_and_preprocessor,
)

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Broj raznih "top-N" podskupova koje poredimo sa punim skupom atributa.
TOP_N_VALUES = [5, 10, 15]

# Mapiranje ime modela -> naziv fajla (isti pattern kao u hyperparameter_tunning.py)
TUNED_MODEL_FILES = {
    "Logistic Regression": "tuned_logistic_regression.joblib",
    "Decision Tree":       "tuned_decision_tree.joblib",
    "Random Forest":       "tuned_random_forest.joblib",
    "Gradient Boosting":   "tuned_gradient_boosting.joblib",
}


def load_all_tuned_models():
    """Učitava sva 4 tuned modela. Vraća dict {ime: model}."""
    models = {}
    for name, fname in TUNED_MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            print(f"Učitavanje: {path}")
            models[name] = joblib.load(path)
        else:
            print(f"⚠️  Nije pronađen: {path} — preskačem {name}")
    if not models:
        print("❌ Nijedan tuned model nije pronađen! Prvo pokrenite hyperparameter_tunning.py")
    return models


def encoded_to_raw(feature):
    """
    Konvertuje enkodirani naziv atributa u originalni naziv kolone.
    Primer: categorical__job_admin. -> job ; numeric__age -> age
    """
    if feature.startswith("numeric__"):
        return feature.replace("numeric__", "")
    if feature.startswith("categorical__"):
        feature = feature.replace("categorical__", "")

    for prefix in CATEGORICAL_FEATURES:
        if feature.startswith(prefix + "_"):
            return prefix

    return feature


def get_feature_importance(model):
    """
    Izvlači feature importance iz modela.
    - Stabla/ansambli (DecisionTree, RandomForest, GradientBoosting): Gini importance.
    - Logistic Regression: apsolutna vrednost koeficijenata.
    Radi sa celim ImbPipeline-om (preprocessor+smote+classifier).
    """
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
    else:
        raise ValueError(
            "Model mora biti ceo Pipeline (sa korakom 'preprocessor' i 'classifier') "
            "da bi se feature importance moglo povezati sa originalnim atributima."
        )

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        raise ValueError("Ovaj model nema direktno dostupnu feature importance vrednost.")

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance_df, model_name, top_n=20):
    """Prikaz feature importance (top_n enkodiranih atributa) za jedan model."""
    top = importance_df.head(top_n).copy()
    top["feature_display"] = top["feature"].apply(
        lambda x: x.replace("categorical__", "").replace("numeric__", "").replace("_", " ").title()
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature_display"][::-1], top["importance"][::-1])
    plt.title(f"Feature importance ({model_name}) - Top {top_n} atributa")
    plt.xlabel("Importance")
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"feature_importance_{fname}.png", dpi=150)
    plt.close()

    print(f"\nTop {top_n} atributa po feature importance ({model_name}):")
    for i, row in top.iterrows():
        raw = encoded_to_raw(row["feature"])
        print(f"  {i + 1}. {row['feature']} -> {raw} ({row['importance']:.4f})")


def plot_all_importances_combined(all_importance, top_n=15):
    """
    Subplot grid: feature importance za sva 4 modela u jednoj figuri.
    Svaki subplot prikazuje top_n enkodiranih atributa za taj model.
    """
    n_models = len(all_importance)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for ax, (model_name, importance_df) in zip(axes, all_importance.items()):
        top = importance_df.head(top_n).copy()
        top["feature_display"] = top["feature"].apply(
            lambda x: x.replace("categorical__", "").replace("numeric__", "").replace("_", " ").title()
        )
        ax.barh(top["feature_display"][::-1], top["importance"][::-1])
        ax.set_title(f"{model_name} — Top {top_n}")
        ax.set_xlabel("Importance")

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Feature importance — svi modeli (Top {top_n})", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_all_models.png", dpi=150)
    plt.close()
    print(f"\n✅ Kombinovani grafik sačuvan: {FIGURES_DIR / 'feature_importance_all_models.png'}")


def get_top_raw_features(importance_df, n):
    """
    Vraća listu od n originalnih (sirovih) naziva kolona, sortiranih po
    najvecem pojedinacnom importance-u medju njihovim enkodiranim kategorijama.
    Npr. ako su 'poutcome_success' i 'poutcome_failure' oba visoko rangirani,
    'poutcome' se u finalnu listu ubraja samo jednom.
    """
    raw_features = []
    for _, row in importance_df.iterrows():
        raw = encoded_to_raw(row["feature"])
        if raw not in raw_features:
            raw_features.append(raw)
        if len(raw_features) >= n:
            break
    return raw_features


def evaluate_classifier(model, X, y):
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
    }


def train_and_evaluate_subset(classifier_template, selected_features, X_train, y_train, X_test, y_test, label):
    """
    Trenira svez pipeline (preprocessor+SMOTE+classifier) na zadatom podskupu
    originalnih (sirovih) atributa i evaluira na test skupu. Koristi se isti
    tip klasifikatora (iste hiperparametre) kao polazni tuned model, da bi
    poređenje "svi vs top-N" bilo fer.
    """
    cat_subset = [c for c in CATEGORICAL_FEATURES if c in selected_features]
    num_subset = [c for c in NUMERIC_FEATURES if c in selected_features]

    subset_preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_subset),
        ("numeric", "passthrough", num_subset),
    ])

    pipeline = ImbPipeline(steps=[
        ("preprocessor", subset_preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier_template),
    ])

    cols = cat_subset + num_subset
    pipeline.fit(X_train[cols], y_train)
    metrics = evaluate_classifier(pipeline, X_test[cols], y_test)
    metrics["label"] = label
    metrics["n_features"] = len(cols)
    metrics["features"] = ", ".join(cols)
    return metrics


def compare_full_vs_topn_for_model(model_name, tuned_model, importance_df, X_train, y_train, X_test, y_test):
    """
    Poredi performanse jednog modela treniranog na SVIM atributima sa
    performansama istog tipa modela treniranog na top-5, top-10, top-15
    atributima (po feature importance tog modela).
    Klasifikator se izvlači iz tuned_model pipeline-a (isti hiperparametri
    kao nakon GridSearchCV), pa je poređenje fer.
    """
    import sklearn.base as skbase

    print(f"\n{'=' * 60}")
    print(f"POREĐENJE: SVI vs. TOP-N — {model_name}")
    print("=" * 60)

    # Koristimo clone tuned klasifikatora — isti hiperparametri, ali svez fit
    classifier_template = skbase.clone(tuned_model.named_steps["classifier"])
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    rows = []

    full_metrics = train_and_evaluate_subset(
        classifier_template, all_features, X_train, y_train, X_test, y_test,
        label="Svi atributi"
    )
    rows.append(full_metrics)
    print(f"\nSvi atributi ({full_metrics['n_features']}): "
          f"F2={full_metrics['f2_yes']:.4f} F1={full_metrics['f1_yes']:.4f} "
          f"Recall={full_metrics['recall']:.4f} Precision={full_metrics['precision']:.4f}")

    for n in TOP_N_VALUES:
        top_features = get_top_raw_features(importance_df, n)
        metrics = train_and_evaluate_subset(
            classifier_template, top_features, X_train, y_train, X_test, y_test,
            label=f"Top {n}"
        )
        rows.append(metrics)
        print(f"\nTop {n} ({metrics['n_features']} atributa: {metrics['features']}): "
              f"F2={metrics['f2_yes']:.4f} F1={metrics['f1_yes']:.4f} "
              f"Recall={metrics['recall']:.4f} Precision={metrics['precision']:.4f}")

    return pd.DataFrame(rows)


def plot_comparison(comparison_df, model_name):
    """Bar plot: F2, F1, Recall, Precision za 'Svi atributi' i svaki Top-N."""
    metrics_to_plot = ["f2_yes", "f1_yes", "recall", "precision"]
    labels = ["F2", "F1", "Recall", "Precision"]

    x = range(len(comparison_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        offset = (i - len(metrics_to_plot) / 2) * width
        ax.bar([p + offset for p in x], comparison_df[metric], width, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["label"], rotation=0)
    ax.set_ylabel("Vrednost metrike")
    ax.set_title(f"Poređenje: svi atributi vs. top-N atributa ({model_name})")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"feature_selection_comparison_{fname}.png", dpi=150)
    plt.close()


def plot_topn_comparison_all_models(all_comparisons):
    """
    Subplot grid: poređenje svi vs top-N za sva 4 modela u jednoj figuri (F2 metrika).
    Svaki subplot = jedan model, x-osa = Svi/Top5/Top10/Top15.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (model_name, comp_df) in zip(axes, all_comparisons.items()):
        ax.bar(comp_df["label"], comp_df["f2_yes"], color="steelblue")
        ax.set_title(model_name)
        ax.set_ylabel("F2 (yes)")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(ax.patches, comp_df["f2_yes"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    for j in range(len(all_comparisons), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Svi atributi vs. Top-N — F2 po modelu", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_selection_comparison_all_models.png", dpi=150)
    plt.close()
    print(f"\n✅ Kombinovani grafik poređenja sačuvan: {FIGURES_DIR / 'feature_selection_comparison_all_models.png'}")


def main():
    print("=" * 60)
    print("FAZA 6: SELEKCIJA NAJZNAČAJNIJIH ATRIBUTA")
    print("=" * 60)

    splits_path = MODEL_DIR / "data_splits.joblib"
    if not splits_path.exists():
        print("❌ data_splits.joblib nije pronađen! Prvo pokrenite model_training.py")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(splits_path)

    tuned_models = load_all_tuned_models()
    if not tuned_models:
        return

    all_importance = {}
    all_comparisons = {}

    for model_name, model in tuned_models.items():
        print(f"\n{'#' * 60}")
        print(f"MODEL: {model_name}")
        print("#" * 60)

        # Feature importance
        importance_df = get_feature_importance(model)
        importance_df["model"] = model_name
        all_importance[model_name] = importance_df

        fname = model_name.lower().replace(" ", "_")
        importance_df.to_csv(RESULTS_DIR / f"feature_importance_{fname}.csv", index=False)
        print(f"✅ Feature importance sačuvan: {RESULTS_DIR / f'feature_importance_{fname}.csv'}")

        plot_feature_importance(importance_df, model_name, top_n=20)

        # Poređenje svi vs top-N
        comp_df = compare_full_vs_topn_for_model(
            model_name, model, importance_df, X_train, y_train, X_test, y_test
        )
        comp_df["model"] = model_name
        all_comparisons[model_name] = comp_df

        comp_df.to_csv(RESULTS_DIR / f"feature_selection_comparison_{fname}.csv", index=False)
        print(f"✅ Poređenje sačuvano: {RESULTS_DIR / f'feature_selection_comparison_{fname}.csv'}")

        plot_comparison(comp_df, model_name)

    # Kombinovani grafici za sve modele
    plot_all_importances_combined(all_importance, top_n=15)
    plot_topn_comparison_all_models(all_comparisons)

    # Jedan objedinjeni CSV sa svim importance vrednostima
    combined_importance = pd.concat(all_importance.values(), ignore_index=True)
    combined_importance.to_csv(RESULTS_DIR / "feature_importance_all_models.csv", index=False)
    print(f"\n✅ Objedinjeni feature importance sačuvan: {RESULTS_DIR / 'feature_importance_all_models.csv'}")

    combined_comparisons = pd.concat(all_comparisons.values(), ignore_index=True)
    combined_comparisons.to_csv(RESULTS_DIR / "feature_selection_comparison_all_models.csv", index=False)
    print(f"✅ Objedinjeno poređenje sačuvano: {RESULTS_DIR / 'feature_selection_comparison_all_models.csv'}")

    print("\n" + "=" * 60)
    print("SELEKCIJA ATRIBUTA ZAVRŠENA")
    print("=" * 60)
    print(f"📁 Rezultati sačuvani u: {RESULTS_DIR}")
    print(f"📊 Grafikoni sačuvani u: {FIGURES_DIR}")


if __name__ == "__main__":
    main()