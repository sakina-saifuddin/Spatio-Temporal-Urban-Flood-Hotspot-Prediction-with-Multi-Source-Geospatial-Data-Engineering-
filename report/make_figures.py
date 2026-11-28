"""Build the report's figures from the project's saved results (no model is retrained).

Run from the repository root:  python report/make_figures.py
Inputs:  reports/results.csv, reports/shap_feature_importance.csv, web/data/repeat.json,
         reports/figures/{mlp,cnn}_training_loss.png, reports/figures/example_tiles.png
Outputs: report/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, PURPLE = "#1F5FC4", "#E07B22", "#5B21B6"

# ---------- Figure: rainfall ablation (storm held out) ----------
r = pd.read_csv(ROOT / "reports" / "results.csv")
models = [("logistic_regression", "LogReg"), ("random_forest", "RF"), ("xgboost", "XGB"),
          ("lightgbm", "LGBM"), ("mlp", "MLP")]
loso = r[r.split == "leave_one_storm_out"].set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, metric, title in zip(axes, ["recall", "roc_auc"], ["Recall", "ROC-AUC"]):
    x = range(len(models))
    with_rain = [loso.loc[m, metric] for m, _ in models]
    no_rain = [loso.loc[f"{m}_no_rain", metric] for m, _ in models]
    ax.bar([i - 0.2 for i in x], with_rain, 0.4, color=ORANGE, label="with rainfall")
    ax.bar([i + 0.2 for i in x], no_rain, 0.4, color=BLUE, label="without rainfall")
    ax.set_xticks(list(x)); ax.set_xticklabels([n for _, n in models])
    ax.set_ylim(0.4 if metric == "recall" else 0.7, 0.95)
    ax.set_title(f"{title}, leave-one-storm-out")
    ax.grid(axis="y", alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
fig.savefig(OUT / "rain_ablation.png"); plt.close(fig)

# ---------- Figure: SHAP importance + repeat-flooding staircase ----------
shap = pd.read_csv(ROOT / "reports" / "shap_feature_importance.csv").sort_values("mean_abs_shap")
nice = {"elevation_m": "Elevation", "paved_pct": "Highway proximity", "slope_degrees": "Slope",
        "distance_to_bayou_m": "Distance to bayou"}
labels = [nice.get(f, f.replace("land cover type: Developed, ", "Land cover: ")) for f in shap.feature]
rep = json.loads((ROOT / "web" / "data" / "repeat.json").read_text(encoding="utf-8"))["by_count"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
a.barh(labels, shap.mean_abs_shap, color=BLUE)
a.set_xlabel("mean |SHAP| (log-odds)"); a.set_title("(a) LightGBM feature importance")
a.grid(axis="x", alpha=0.3)
cols = ["#D9DEE7", "#DDD6FE", "#A78BFA", "#7C3AED", "#4C1D95"]
xs = [str(x["storms"]) for x in rep]
b.bar(xs, [x["avg_score"] for x in rep], color=cols)
for i, x in enumerate(rep):
    b.text(i, x["avg_score"] + 0.015, f"{x['avg_score']:.2f}", ha="center", fontsize=8)
    b.text(i, 0.015, f"n={x['areas']:,}", ha="center", fontsize=6.5, color="white" if i >= 3 else "#333")
b.set_xlabel("storms with 311 reports in the area (of 4)"); b.set_ylabel("mean model score")
b.set_ylim(0, 0.75); b.set_title("(b) Model score vs. repeat flooding")
fig.tight_layout(); fig.savefig(OUT / "shap_repeat.png"); plt.close(fig)

# ---------- Figure: training losses side by side (existing project plots) ----------
ims = [Image.open(ROOT / "reports" / "figures" / f).convert("RGB")
       for f in ("mlp_training_loss.png", "cnn_training_loss.png")]
h = 700
ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
canvas = Image.new("RGB", (sum(im.width for im in ims) + 30, h), "white")
x = 0
for im in ims:
    canvas.paste(im, (x, 0)); x += im.width + 30
canvas.save(OUT / "training_loss.png")

# ---------- Copy the CNN input examples ----------
Image.open(ROOT / "reports" / "figures" / "example_tiles.png").convert("RGB").save(OUT / "example_tiles.png")
print("figures written to", OUT)
