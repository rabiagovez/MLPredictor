# -*- coding: utf-8 -*-
"""
generate_report.py - Model Sonuclarini Gorsellestirir
Ciktı: reports/ klasorune PNG dosyalari
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, joblib

MODELS_DIR = "models_live"
REPORTS_DIR = "reports_live"
os.makedirs(REPORTS_DIR, exist_ok=True)

COLORS = {
    "Random Forest": "#2ecc71",
    "LightGBM":      "#3498db",
    "Stacking":      "#e74c3c",
    "Dynamic Ensemble": "#f1c40f",
}
BG = "#0f1117"
CARD = "#1a1d2e"
TEXT = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       "#2a2a3e",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


def load_results():
    metrics = pd.read_csv(os.path.join(MODELS_DIR, "metrics.csv"), encoding="utf-8-sig")
    pred_candidates = [
        os.path.join(MODELS_DIR, "test_predictions_with_dynamic_ensemble.csv"),
        "predictions_clean_2026.csv",
    ]
    pred_path = next((p for p in pred_candidates if os.path.exists(p)), None)
    if pred_path is None:
        raise FileNotFoundError("[!] Tahmin dosyasi bulunamadi.")
    preds = pd.read_csv(pred_path, encoding="utf-8-sig")

    # Farkli pipeline ciktilarini ortak formata getir.
    rename_map = {}
    if "urun_adi" in preds.columns and "urun" not in preds.columns:
        rename_map["urun_adi"] = "urun"
    if "pred_random_forest" in preds.columns:
        rename_map["pred_random_forest"] = "tahmin_random_forest"
    if "pred_lightgbm" in preds.columns:
        rename_map["pred_lightgbm"] = "tahmin_lightgbm"
    if "pred_stacking" in preds.columns:
        rename_map["pred_stacking"] = "tahmin_stacking"
    if "pred_dynamic_ensemble" in preds.columns:
        rename_map["pred_dynamic_ensemble"] = "tahmin_dynamic_ensemble"
    if rename_map:
        preds = preds.rename(columns=rename_map)

    fi_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    fi      = pd.read_csv(fi_path, encoding="utf-8-sig") if os.path.exists(fi_path) else None
    preds["hafta_baslangic"] = pd.to_datetime(preds["hafta_baslangic"])
    return metrics, preds, fi


def pretty_model_name_from_col(col):
    key = col.replace("tahmin_", "").lower()
    mapping = {
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "stacking": "Stacking",
        "dynamic_ensemble": "Dynamic Ensemble",
    }
    return mapping.get(key, col.replace("tahmin_", "").replace("_", " ").title())


# ─── 1. METRIK KARSILASTIRMA BAR CHART ────────────────────────────────────────

def plot_metrics(metrics):
    metric_cols = [
        c for c in ["MAE", "RMSE", "wMAPE(%)", "MDA", "R2"]
        if c in metrics.columns
    ]
    n = len(metric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    fig.suptitle("Model Basari Metrikleri Karsilastirmasi (Test: 2026)",
                 fontsize=16, fontweight="bold", color=TEXT, y=1.01)
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(metric_cols):
        ax = axes[i]
        models = metrics["Model"].tolist()
        vals   = metrics[col].tolist()
        bar_colors = [COLORS.get(m, "#aaa") for m in models]
        bars = ax.bar(models, vals, color=bar_colors, width=0.5, edgecolor="#111", linewidth=0.8)

        # Deger etiketi
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.02,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=9, color=TEXT, fontweight="bold")

        ax.set_title(col, fontsize=13, color=TEXT, pad=8)
        ax.set_ylim(0, max(v for v in vals if v) * 1.2 if any(v for v in vals if v) else 1)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "1_metrik_karsilastirma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── 2. GERCEK vs TAHMIN SCATTER ──────────────────────────────────────────────

def plot_actual_vs_predicted(preds):
    tahmin_cols = [c for c in preds.columns if c.startswith("tahmin_") and "ortalama" not in c]
    n = len(tahmin_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Gercek Fiyat vs Tahmin Fiyat (2026 Test Verisi)",
                 fontsize=15, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, tahmin_cols):
        model_name = pretty_model_name_from_col(col)
        color = COLORS.get(model_name, "#aaa")
        y_true = preds["hedef_haftalik"].values
        y_pred = preds[col].values
        mask   = ~(np.isnan(y_true) | np.isnan(y_pred))

        ax.scatter(y_true[mask], y_pred[mask], alpha=0.4, s=18,
                   color=color, edgecolors="none")
        lims = [min(y_true[mask].min(), y_pred[mask].min()),
                max(y_true[mask].max(), y_pred[mask].max())]
        ax.plot(lims, lims, "w--", linewidth=1.5, alpha=0.7, label="Ideal")
        ax.set_xlabel("Gercek Fiyat (TL)", fontsize=11)
        ax.set_ylabel("Tahmin Fiyat (TL)", fontsize=11)
        ax.set_title(model_name, fontsize=13, color=color)
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "2_gercek_vs_tahmin.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── 3. REZIDUAL DAGILIM HISTOGRAM ────────────────────────────────────────────

def plot_residuals(preds):
    tahmin_cols = [c for c in preds.columns if c.startswith("tahmin_") and "ortalama" not in c]
    fig, axes = plt.subplots(1, len(tahmin_cols), figsize=(6 * len(tahmin_cols), 5))
    if len(tahmin_cols) == 1:
        axes = [axes]

    fig.suptitle("Rezidual Dagilim (Gercek - Tahmin)",
                 fontsize=15, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, tahmin_cols):
        model_name = pretty_model_name_from_col(col)
        color = COLORS.get(model_name, "#aaa")
        residuals = preds["hedef_haftalik"] - preds[col]
        residuals = residuals.dropna()

        ax.hist(residuals, bins=50, color=color, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="white", linestyle="--", linewidth=1.5, label="Sifir")
        ax.axvline(residuals.mean(), color="yellow", linestyle="-", linewidth=1.5,
                   label=f"Ort={residuals.mean():.2f}")
        ax.set_xlabel("Rezidual (TL)", fontsize=11)
        ax.set_ylabel("Frekans", fontsize=11)
        ax.set_title(model_name, fontsize=13, color=color)
        ax.legend(fontsize=9)
        ax.grid(True)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "3_rezidual_dagilim.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── 4. ZAMAN SERİSİ: GERCEK vs 3 MODEL (ORNEK URUNLER) ─────────────────────

def plot_timeseries(preds, top_n_products=4):
    products = preds.groupby("urun")["hedef_haftalik"].count().nlargest(top_n_products).index.tolist()
    tahmin_cols = [c for c in preds.columns if c.startswith("tahmin_") and "ortalama" not in c]

    fig, axes = plt.subplots(top_n_products, 1, figsize=(16, 4 * top_n_products))
    if top_n_products == 1:
        axes = [axes]
    fig.suptitle("Zaman Serisi: Gercek vs Model Tahminleri (2026)",
                 fontsize=15, fontweight="bold", color=TEXT)

    for ax, urun in zip(axes, products):
        sub = preds[preds["urun"] == urun].sort_values("hafta_baslangic")
        ax.plot(sub["hafta_baslangic"], sub["hedef_haftalik"], "w-o", markersize=3,
                linewidth=1.8, label="Gercek", zorder=5)
        for col in tahmin_cols:
            model_name = pretty_model_name_from_col(col)
            color = COLORS.get(model_name, "#aaa")
            ax.plot(sub["hafta_baslangic"], sub[col], "-", color=color,
                    linewidth=1.4, alpha=0.85, label=model_name)
        ax.set_title(urun, fontsize=12, color="#f0c040")
        ax.set_ylabel("Fiyat (TL)", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True)

    axes[-1].set_xlabel("Tarih", fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "4_zaman_serisi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── 5. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def plot_feature_importance(fi):
    if fi is None or fi.empty:
        print("Feature importance verisi yok, atlaniyor.")
        return
    fi_models = fi["Model"].unique()
    n = len(fi_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Feature Importance (Top 15)", fontsize=15, fontweight="bold", color=TEXT)

    for ax, model_name in zip(axes, fi_models):
        sub = fi[fi["Model"] == model_name].nlargest(15, "Importance")
        color = COLORS.get(model_name, "#aaa")
        ax.barh(sub["Feature"], sub["Importance"], color=color, edgecolor="#111")
        ax.invert_yaxis()
        ax.set_title(model_name, fontsize=13, color=color)
        ax.set_xlabel("Onem Skoru", fontsize=11)
        ax.grid(axis="x")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "5_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── 6. METRİK OZET TABLOSU (PNG) ────────────────────────────────────────────

def plot_metrics_table(metrics):
    metric_label_map = {
        "MAE": "MAE (TL)",
        "RMSE": "RMSE (TL)",
        "wMAPE(%)": "wMAPE (%)",
        "MDA": "MDA",
        "R2": "R2",
    }
    metric_cols = [c for c in metric_label_map if c in metrics.columns]
    fig_w = max(12, 3 + 1.6 * (1 + len(metric_cols)))
    fig, ax = plt.subplots(figsize=(fig_w, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    col_labels = ["Model"] + [metric_label_map[c] for c in metric_cols]
    rows = []
    for _, row in metrics.iterrows():
        vals = [row["Model"]]
        for c in metric_cols:
            v = row[c]
            if c in ["wMAPE(%)"] and pd.notna(v):
                vals.append(str(v) + "%")
            else:
                vals.append(str(v))
        rows.append(vals)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(CARD if r > 0 else "#2a2d4e")
        cell.set_edgecolor("#444")
        cell.set_text_props(color=TEXT, fontweight="bold" if r == 0 else "normal")
        if r > 0 and c == 0:
            model = rows[r - 1][0]
            cell.set_facecolor(COLORS.get(model, CARD))
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Konya Hal Fiyatlari - Model Performans Ozeti (Test: 2026)",
                 fontsize=14, color=TEXT, pad=20, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "0_metrik_ozet_tablosu.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Kaydedildi: " + path)


# ─── ANA ─────────────────────────────────────────────────────────────────────

def main():
    print("[*] Raporlar olusturuluyor...")
    metrics, preds, fi = load_results()

    plot_metrics_table(metrics)
    plot_metrics(metrics)
    plot_actual_vs_predicted(preds)
    plot_residuals(preds)
    plot_timeseries(preds)
    plot_feature_importance(fi)

    print("\nTum raporlar hazir: " + REPORTS_DIR + "/")


if __name__ == "__main__":
    main()
