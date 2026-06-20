import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
)

from data_preparation import MODEL_DIR, RESULTS_DIR, FIGURES_DIR

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]

MIN_ACCEPTABLE_PRECISION = 0.20

# Opseg pragova koji se ispituje pri threshold tuningu.
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def load_models():
    """Učitavanje svih istreniranih modela"""
    models = {}
    for model_name in MODEL_NAMES:
        fname = model_name.lower().replace(" ", "_")

        path = MODEL_DIR / f"tuned_{fname}.joblib"
        if path.exists():
            models[model_name] = joblib.load(path)
            print(f"✅ Učitan tuned model: {model_name}")
            continue

        path = MODEL_DIR / f"{fname}.joblib"
        if path.exists():
            models[model_name] = joblib.load(path)
            print(f"✅ Učitan model: {model_name}")
            continue

    if not models:
        best_paths = [
            MODEL_DIR / "best_tuned_model.joblib",
            MODEL_DIR / "best_model.joblib",
            MODEL_DIR / "final_model.joblib",
        ]
        for path in best_paths:
            if path.exists():
                model = joblib.load(path)
                models["Best Model"] = model
                print(f"✅ Učitan najbolji model: {path}")
                break

    if not models:
        print("⚠️  Nijedan model nije pronađen!")

    return models


def evaluate_model(model, X, y):
    """Evaluacija pojedinačnog modela na default pragu (0.5)"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1_yes": f1_score(y, y_pred, pos_label=1),
        "f2_yes": fbeta_score(y, y_pred, beta=2, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        # PR-AUC je informativniji od ROC-AUC za retku pozitivnu klasu (~11%
        # "yes"); baseline za PR-AUC je udeo pozitivne klase (~0.11), za
        # razliku od ROC-AUC ciji je baseline uvek 0.5.
        "pr_auc": average_precision_score(y, y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y, y_pred, target_names=["no", "yes"], zero_division=0
        ),
    }


def find_best_threshold(y_val, y_val_prob, metric="f2"):
    """
    Trazi prag (na VALIDACIONOM skupu) koji maksimizuje zadatu metriku
    (podrazumevano F2, isti kriterijum kao izbor modela).

    Threshold tuning menja samo granicu odluke (model.predict koristi 0.5
    podrazumevano) - ne menja ni model ni podatke. Bira se iskljucivo na
    validation skupu da bi se izbeglo "curenje" informacije o test skupu u
    odluku o pragu; nakon izbora, isti prag se direktno primenjuje na test.
    """
    best_threshold = 0.5
    best_score = -1.0
    rows = []

    for t in THRESHOLD_GRID:
        y_pred = (y_val_prob >= t).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1, "f2": f2})

        if metric == "f2":
            score = f2
        elif metric == "f1":
            score = f1
        else:
            score = r
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, pd.DataFrame(rows)


def evaluate_all(models, X_test, y_test):
    """Evaluacija svih modela na default pragu (0.5)"""
    rows = []
    detailed = {}

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        metrics = evaluate_model(model, X_test, y_test)
        detailed[model_name] = metrics

        print("Test Accuracy:", round(metrics["accuracy"], 4))
        print("Test Precision:", round(metrics["precision"], 4))
        print("Test Recall:", round(metrics["recall"], 4))
        print("Test F1 (yes):", round(metrics["f1_yes"], 4))
        print("Test F2 (yes):", round(metrics["f2_yes"], 4))
        print("Test ROC-AUC:", round(metrics["roc_auc"], 4))
        print("Test PR-AUC:", round(metrics["pr_auc"], 4))
        print("\nConfusion matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification report:")
        print(metrics["classification_report"])

        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_yes": metrics["f1_yes"],
            "f2_yes": metrics["f2_yes"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "tn": metrics["tn"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tp": metrics["tp"],
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)

    print("\n" + "=" * 60)
    print("POREDJENJE MODELA NA TEST SKUPU (prag=0.5)")
    print("=" * 60)
    print(results_df.sort_values(by="f2_yes", ascending=False).to_string(index=False))

    return results_df, detailed


def threshold_tuning_all(models, X_val, y_val, X_test, y_test):
    """
    Za svaki model: nadje optimalan prag na validaciji (max F2), zatim
    primeni taj prag na test skup. Prikazuje default (0.5) i optimizovani
    rezultat jedno pored drugog (Accuracy, Precision, Recall, F1, F2) radi
    transparentnosti i "balansiranog" pregleda - F2 odredjuje izbor praga,
    ali se sve metrike vide zajedno.
    """
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING (optimizacija praga odluke)")
    print("=" * 60)
    print("Prag se bira na VALIDACIONOM skupu (max F2), a primenjuje na TEST.\n")

    rows = []
    curves = {}

    for model_name, model in models.items():
        y_val_prob = model.predict_proba(X_val)[:, 1]
        best_threshold, curve_df = find_best_threshold(y_val, y_val_prob, metric="f2")
        curves[model_name] = curve_df

        y_test_prob = model.predict_proba(X_test)[:, 1]

        # Default prag (0.5) na test skupu
        y_pred_default = (y_test_prob >= 0.5).astype(int)
        default_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_default),
            "precision": precision_score(y_test, y_pred_default, zero_division=0),
            "recall": recall_score(y_test, y_pred_default, zero_division=0),
            "f1": f1_score(y_test, y_pred_default, zero_division=0),
            "f2": fbeta_score(y_test, y_pred_default, beta=2, zero_division=0),
        }

        # Optimizovani prag (izabran na validaciji) na test skupu
        y_pred_opt = (y_test_prob >= best_threshold).astype(int)
        opt_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_opt),
            "precision": precision_score(y_test, y_pred_opt, zero_division=0),
            "recall": recall_score(y_test, y_pred_opt, zero_division=0),
            "f1": f1_score(y_test, y_pred_opt, zero_division=0),
            "f2": fbeta_score(y_test, y_pred_opt, beta=2, zero_division=0),
        }

        print(f"\n{model_name}")
        print(f"  Optimalni prag (iz validacije): {best_threshold:.2f}")
        print(f"  TEST @ prag 0.50:  Acc={default_metrics['accuracy']:.4f} "
              f"P={default_metrics['precision']:.4f} R={default_metrics['recall']:.4f} "
              f"F1={default_metrics['f1']:.4f} F2={default_metrics['f2']:.4f}")
        print(f"  TEST @ prag {best_threshold:.2f}:  Acc={opt_metrics['accuracy']:.4f} "
              f"P={opt_metrics['precision']:.4f} R={opt_metrics['recall']:.4f} "
              f"F1={opt_metrics['f1']:.4f} F2={opt_metrics['f2']:.4f}")
        print(f"  Promena F2: {opt_metrics['f2'] - default_metrics['f2']:+.4f}")
        if opt_metrics["precision"] < MIN_ACCEPTABLE_PRECISION:
            print(f"  ⚠️  UPOZORENJE: Precision na optimizovanom pragu ({opt_metrics['precision']:.4f}) "
                  f"je ispod prihvatljivog minimuma ({MIN_ACCEPTABLE_PRECISION}).")

        rows.append({
            "model": model_name,
            "best_threshold": best_threshold,
            "default_accuracy": default_metrics["accuracy"],
            "default_precision": default_metrics["precision"],
            "default_recall": default_metrics["recall"],
            "default_f1": default_metrics["f1"],
            "default_f2": default_metrics["f2"],
            "optimized_accuracy": opt_metrics["accuracy"],
            "optimized_precision": opt_metrics["precision"],
            "optimized_recall": opt_metrics["recall"],
            "optimized_f1": opt_metrics["f1"],
            "optimized_f2": opt_metrics["f2"],
            "f2_improvement": opt_metrics["f2"] - default_metrics["f2"],
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(RESULTS_DIR / "threshold_tuning_results.csv", index=False)
    print(f"\n✅ Rezultati threshold tuninga sačuvani u: {RESULTS_DIR / 'threshold_tuning_results.csv'}")

    plot_threshold_curves(curves)
    plot_threshold_comparison(threshold_df)

    return threshold_df


def plot_threshold_curves(curves):
    """Prikaz Precision/Recall/F1/F2 u funkciji praga, za svaki model."""
    n_models = len(curves)
    cols = 2
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5 * rows))
    # plt.subplots uvek vraca numpy.ndarray cim je rows>1 ili cols>1 (cols je
    # ovde fiksno 2), pa cak i za n_models=1 imamo niz od 2 Axes-a (drugi
    # ostaje prazan/sakriven). Samo za n_models=1 I cols=1 bi axes bio goli
    # Axes objekat - taj slucaj se ovde ne desava jer je cols uvek 2, ali
    # flatten() je bezbedan poziv u oba slucaja (ndarray.flatten() postoji,
    # a za goli Axes bismo ga eksplicitno umotali u listu).
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (model_name, curve_df) in enumerate(curves.items()):
        ax = axes[i]
        ax.plot(curve_df["threshold"], curve_df["precision"], label="Precision", alpha=0.8)
        ax.plot(curve_df["threshold"], curve_df["recall"], label="Recall", alpha=0.8)
        ax.plot(curve_df["threshold"], curve_df["f1"], label="F1", alpha=0.6, linestyle="--")
        ax.plot(curve_df["threshold"], curve_df["f2"], label="F2", linewidth=2, color="black")
        best_idx = curve_df["f2"].idxmax()
        ax.axvline(curve_df.loc[best_idx, "threshold"], color="gray", linestyle=":", alpha=0.6)
        ax.set_title(model_name)
        ax.set_xlabel("Prag odluke")
        ax.set_ylabel("Vrednost metrike")
        ax.legend()
        ax.grid(alpha=0.3)

    for j in range(len(curves), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Precision / Recall / F1 / F2 u funkciji praga (validacioni skup)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tuning_curves.png", dpi=150)
    plt.close()


def plot_threshold_comparison(threshold_df):
    """Bar plot: F2 pre i posle threshold tuninga, za sve modele."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(threshold_df))
    width = 0.35

    ax.bar(x, threshold_df["default_f2"], width, label="Default prag (0.5)", alpha=0.8)
    ax.bar([i + width for i in x], threshold_df["optimized_f2"], width, label="Optimizovani prag", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Test F2 (yes)")
    ax.set_title("F2 pre i posle threshold tuninga")
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(threshold_df["model"], rotation=15)
    ax.legend()

    for i, row in threshold_df.iterrows():
        ax.text(i, row["default_f2"] + 0.01, f"{row['default_f2']:.3f}", ha="center", fontsize=8)
        ax.text(i + width, row["optimized_f2"] + 0.01, f"{row['optimized_f2']:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tuning_comparison.png", dpi=150)
    plt.close()


def baseline_comparison(X_train, y_train, X_test, y_test):
    """Poređenje sa baseline modelom (DummyClassifier)"""
    print("\n" + "=" * 60)
    print("BASELINE MODEL (DummyClassifier)")
    print("=" * 60)

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_prob = dummy.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Baseline (DummyClassifier)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_yes": f1_score(y_test, y_pred, pos_label=1),
        "f2_yes": fbeta_score(y_test, y_pred, beta=2, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print("Baseline rezultati (stratified):")
    for key, value in metrics.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")

    # Baseline rezultati ranije nisu bili perzistirani ni u jednom results
    # fajlu, sto je onemogucavalo poredjenje "model je bolji od slucajnog
    # pogadjanja" u dokumentaciji bez rucnog kopiranja brojeva iz konzole.
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    print(f"\n✅ Baseline rezultati sačuvani u: {RESULTS_DIR / 'baseline_results.csv'}")

    return metrics


def plot_confusion_matrices(detailed):
    """Prikaz matrica konfuzije za sve modele (prag=0.5)"""
    n_models = len(detailed)
    cols = 2
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    # Vidi napomenu u plot_threshold_curves - cols je fiksno 2, pa axes
    # ostaje numpy.ndarray cak i za n_models=1.
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (model_name, metrics) in enumerate(detailed.items()):
        if i < len(axes):
            sns.heatmap(
                metrics["confusion_matrix"],
                annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "yes"], yticklabels=["no", "yes"],
                ax=axes[i]
            )
            axes[i].set_title(model_name)
            axes[i].set_xlabel("Predviđeno")
            axes[i].set_ylabel("Stvarno")

    for i in range(len(detailed), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    """Prikaz ROC krivih za sve modele"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)
    ax.set_title("ROC krive - svi modeli")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()


def plot_precision_recall_curves(models, X_test, y_test):
    """Prikaz Precision-Recall krivih za sve modele.

    Korisnije od ROC krive kada je pozitivna klasa retka (~11%), jer
    direktno pokazuje trade-off relevantan za izbor praga odluke.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax)

    baseline_rate = y_test.mean()
    ax.axhline(baseline_rate, linestyle="--", color="gray", alpha=0.5, label=f"Baseline ({baseline_rate:.3f})")
    ax.set_title("Precision-Recall krive - svi modeli")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=150)
    plt.close()


def generate_summary_report(results_df, threshold_df, baseline_metrics):
    """Generisanje sažetog izveštaja - štampa u konzolu I čuva u fajl."""
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("\n" + "=" * 60)
    emit("SAŽETAK EVALUACIJE")
    emit("=" * 60)

    best_row = results_df.loc[results_df["f2_yes"].idxmax()]
    best_model_name = best_row["model"]

    emit(f"\n🏆 Najbolji model (po F2, yes klasa, prag=0.5): {best_model_name}")
    emit(f"   - F2 (yes): {best_row['f2_yes']:.4f}")
    emit(f"   - F1 (yes): {best_row['f1_yes']:.4f}")
    emit(f"   - Recall: {best_row['recall']:.4f}")
    emit(f"   - Precision: {best_row['precision']:.4f}")
    emit(f"   - Accuracy: {best_row['accuracy']:.4f}")
    emit(f"   - ROC AUC: {best_row['roc_auc']:.4f}")
    emit(f"   - PR AUC: {best_row['pr_auc']:.4f}")
    emit(f"   - TP: {best_row['tp']}, TN: {best_row['tn']}")
    emit(f"   - FP: {best_row['fp']}, FN: {best_row['fn']}")
    if best_row["precision"] < MIN_ACCEPTABLE_PRECISION:
        emit(f"   ⚠️  Precision je ispod prihvatljivog minimuma ({MIN_ACCEPTABLE_PRECISION}) - "
             f"proveriti poslovnu opravdanost ovog izbora.")

    if threshold_df is not None:
        best_thr_row = threshold_df[threshold_df["model"] == best_model_name].iloc[0]
        emit(f"\n📐 Nakon threshold tuninga (prag={best_thr_row['best_threshold']:.2f}):")
        emit(f"   - F2: {best_thr_row['default_f2']:.4f} → {best_thr_row['optimized_f2']:.4f} "
             f"({best_thr_row['f2_improvement']:+.4f})")
        emit(f"   - F1: {best_thr_row['default_f1']:.4f} → {best_thr_row['optimized_f1']:.4f}")
        emit(f"   - Recall: {best_thr_row['default_recall']:.4f} → {best_thr_row['optimized_recall']:.4f}")
        emit(f"   - Precision: {best_thr_row['default_precision']:.4f} → {best_thr_row['optimized_precision']:.4f}")

    if baseline_metrics:
        emit(f"\n📊 Poređenje sa baseline modelom:")
        emit(f"   - F2 (yes) poboljšanje: {best_row['f2_yes'] - baseline_metrics['f2_yes']:+.4f}")
        emit(f"   - PR-AUC poboljšanje: {best_row['pr_auc'] - baseline_metrics['pr_auc']:+.4f}")
        emit(f"   - ROC AUC poboljšanje: {best_row['roc_auc'] - baseline_metrics['roc_auc']:+.4f}")

    emit(f"\n💡 Zaključak:")
    emit(f"   Tema projekta je predikcija koliko ce klijenata prihvatiti ponudu banke,")
    emit(f"   pa se modeli rangiraju po F2 meri:")
    emit(f"   F2 = 5 * (Precision * Recall) / (4 * Precision + Recall)")
    emit(f"   F2 daje Recall-u 4x vecu tezinu nego Precision-u, jer se propusten")
    emit(f"   klijent koji bi se pretplatio (FN) smatra skupljom greskom od")
    emit(f"   suvisnog poziva (FP) - ali, za razliku od cistog Recall-a, F2 i dalje")
    emit(f"   donekle kaznjava modele sa veoma niskim Precision-om.")
    if best_row['recall'] > 0.6:
        emit("   Model uspesno prepoznaje vecinu klijenata sklonih pretplati (visok Recall).")
    elif best_row['recall'] > 0.4:
        emit("   Model ima umerenu sposobnost prepoznavanja klijenata sklonih pretplati.")
    else:
        emit("   Model propusta znacajan deo klijenata sklonih pretplati.")
    emit(f"   Niz uklonjenog 'duration' atributa (data leakage) i jako neuravnotezenih")
    emit(f"   klasa (~11% pozitivnih), ovo je ocekivan i objasnjiv rezultat za ovaj problem.")

    emit(f"\n📁 Detaljni rezultati sačuvani u: {RESULTS_DIR}")
    emit(f"📊 Grafikoni sačuvani u: {FIGURES_DIR}")

    report_path = RESULTS_DIR / "evaluation_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Izveštaj sačuvan u: {report_path}")


def main():
    """Glavna funkcija za evaluaciju"""
    print("=" * 60)
    print("FAZA 5: EVALUACIJA MODELA")
    print("=" * 60)

    splits_path = MODEL_DIR / "data_splits.joblib"
    if not splits_path.exists():
        print("❌ data_splits.joblib nije pronađen!")
        print("   Prvo pokrenite model_training.py")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(splits_path)
    print(f"✅ Podaci učitani: X_test {X_test.shape}, y_test {y_test.shape}")

    models = load_models()
    if not models:
        print("❌ Nijedan model nije pronađen!")
        return

    baseline_metrics = baseline_comparison(X_train, y_train, X_test, y_test)

    results_df, detailed = evaluate_all(models, X_test, y_test)

    threshold_df = threshold_tuning_all(models, X_val, y_val, X_test, y_test)

    plot_confusion_matrices(detailed)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    print(f"✅ Grafikoni sačuvani u: {FIGURES_DIR}")

    generate_summary_report(results_df, threshold_df, baseline_metrics)

    print("\n" + "=" * 60)
    print("EVALUACIJA ZAVRŠENA")
    print("=" * 60)


if __name__ == "__main__":
    main()