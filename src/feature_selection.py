import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

from data_preparation import (
    DATA_PATH, MODEL_DIR, RESULTS_DIR, FIGURES_DIR,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_data_and_preprocessor,
)
from model_training import build_pipeline, get_models

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Broj raznih "top-N" podskupova koje poredimo sa punim skupom atributa.
TOP_N_VALUES = [5, 10, 15]


def load_best_model():
    """Učitavanje najboljeg modela iz više mogućih lokacija (tuning ima prioritet)."""
    model_paths = [
        MODEL_DIR / "best_tuned_model.joblib",
        MODEL_DIR / "best_model.joblib",
       
    ]
    for path in model_paths:
        if path.exists():
            print(f"Učitavanje modela: {path}")
            return joblib.load(path)

    model_files = list(MODEL_DIR.glob("best_model*.joblib"))
    if model_files:
        print(f"Učitavanje modela: {model_files[0]}")
        return joblib.load(model_files[0])

    print("⚠️  Nijedan model nije pronađen!")
    return None


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
    Radi i sa golim klasifikatorom i sa celim ImbPipeline-om (preprocessor+smote+classifier).
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
    """Prikaz feature importance (top_n enkodiranih atributa)."""
    top = importance_df.head(top_n).copy()
    top["feature_display"] = top["feature"].apply(
        lambda x: x.replace("categorical__", "").replace("numeric__", "").replace("_", " ").title()
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature_display"][::-1], top["importance"][::-1])
    plt.title(f"Feature importance ({model_name}) - Top {top_n} atributa")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print(f"\nTop {top_n} atributa po feature importance ({model_name}):")
    for i, row in top.iterrows():
        raw = encoded_to_raw(row["feature"])
        print(f"  {i + 1}. {row['feature']} -> {raw} ({row['importance']:.4f})")


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
    Trenira SVEZ pipeline (preprocessor+SMOTE+classifier) na zadatom podskupu
    originalnih (sirovih) atributa i evaluira na test skupu. Koristi se isti
    tip klasifikatora (iste hiperparametre) kao najbolji model, da bi
    poređenje "svi vs top-N" bilo fer (razlika dolazi samo od atributa, ne
    od drugog algoritma).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

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


def compare_full_vs_topn(best_model_name, importance_df, X_train, y_train, X_test, y_test):
    """
    Poredi performanse modela treniranog na SVIM atributima sa performansama
    istog tipa modela treniranog na top-5, top-10, top-15 atributima (po
    feature importance). Klasifikator se ponovo kreira "od nule" sa istim
    hiperparametrima kao polazni model.
    """
    print("\n" + "=" * 60)
    print("POREĐENJE: SVI ATRIBUTI vs. TOP-N ATRIBUTA")
    print("=" * 60)

    classifier_template = get_models()[best_model_name]
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
        print(f"\nTop {n} ({metrics['n_features']} sirovih atributa: {metrics['features']}): "
              f"F2={metrics['f2_yes']:.4f} F1={metrics['f1_yes']:.4f} "
              f"Recall={metrics['recall']:.4f} Precision={metrics['precision']:.4f}")

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_DIR / "feature_selection_comparison.csv", index=False)
    print(f"\n✅ Poređenje sačuvano u: {RESULTS_DIR / 'feature_selection_comparison.csv'}")

    plot_comparison(comparison_df, best_model_name)
    return comparison_df


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
    plt.savefig(FIGURES_DIR / "feature_selection_comparison.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("FAZA 6: SELEKCIJA NAJZNAČAJNIJIH ATRIBUTA")
    print("=" * 60)

    splits_path = MODEL_DIR / "data_splits.joblib"
    if not splits_path.exists():
        print("❌ data_splits.joblib nije pronađen! Prvo pokrenite model_training.py")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(splits_path)

    model = load_best_model()
    if model is None:
        print("❌ Nema modela za analizu. Prvo pokrenite model_training.py")
        return

    best_model_name_path = MODEL_DIR / "best_tuned_model_name.txt"
    if not best_model_name_path.exists():
        best_model_name_path = MODEL_DIR / "best_model_name.txt"
    if best_model_name_path.exists():
        best_model_name = best_model_name_path.read_text(encoding="utf-8").strip()
    else:
        print("⚠️  Ime najboljeg modela nije pronađeno, podrazumevam 'Logistic Regression'.")
        best_model_name = "Logistic Regression"

    print(f"Najbolji model: {best_model_name}")

    importance_df = get_feature_importance(model)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print(f"\n✅ Feature importance sačuvan u: {RESULTS_DIR / 'feature_importance.csv'}")

    plot_feature_importance(importance_df, best_model_name, top_n=20)

    compare_full_vs_topn(best_model_name, importance_df, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("SELEKCIJA ATRIBUTA ZAVRŠENA")
    print("=" * 60)
    print(f"📁 Rezultati sačuvani u: {RESULTS_DIR}")
    print(f"📊 Grafikoni sačuvani u: {FIGURES_DIR}")


if __name__ == "__main__":
    main()