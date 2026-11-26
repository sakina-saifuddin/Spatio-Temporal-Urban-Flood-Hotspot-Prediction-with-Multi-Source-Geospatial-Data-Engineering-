"""Page 5 — Behind the Predictions. Chapter 5: Discover the Science."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts
from components.disclaimers import render_disclaimer_banner, render_limitations, render_official_resources
from components.mascot import scout_says
from components.navigation import chapter_header
from utils.data_loader import figure_path, load_global_shap, load_metadata, load_report_csv, load_storm_metrics
from utils.grid_utils import FEATURE_LABELS

metadata = load_metadata()
model = metadata["model"]
ts = metadata["threshold_selection"]
shap_meta = metadata["shap"]

TEAM = ["Sakina Saifuddin", "Aaliyah D. McAfee", "Van Bao Nguyen"]  # from README.md §12

chapter_header(
    "insights",
    "Behind the Predictions",
    "How the FloodSense model was built, validated, and explained — and where its limits are.",
)
render_disclaimer_banner()
scout_says(
    "This is where I show my work! Everything here comes from the team's saved research results. Nothing on this "
    "page was retrained or re-evaluated for the app.",
    title="Discover the science",
)

tabs = st.tabs(["Purpose & data", "Model & validation", "Independent test", "Tuning", "Rainfall", "SHAP", "Limitations", "Team"])

# --- Purpose & data --------------------------------------------------------
with tabs[0]:
    st.subheader("Project purpose")
    st.markdown(
        "The research asks whether historical Houston 311 flood reports, combined with terrain, drainage, land-cover, and "
        "highway-proximity data, can help identify locations associated with flooding in a storm that was not used for "
        "training. FloodSense makes the frozen model's saved outputs explorable and explainable."
    )
    st.subheader("Data sources")
    st.dataframe(pd.DataFrame([
        ("Flood reports and sampled no-report locations", "City of Houston 311 records", "Training labels and location sample"),
        ("Elevation and slope", "USGS 1-meter Digital Elevation Model via The National Map", "`elevation_m`, `slope_degrees`"),
        ("Streams and bayous", "USGS National Hydrography Dataset", "`distance_to_bayou_m`"),
        ("Land cover", "NLCD", "`land_cover_type`"),
        ("Highway proximity", "OpenStreetMap", "`paved_pct` (share of tract near major highways)"),
        ("Rainfall", "NOAA GHCN-Daily", "Used in earlier comparisons; excluded from the final model"),
    ], columns=["Data", "Source", "Use"]), hide_index=True, width="stretch")
    st.subheader("Environmental features")
    for name, text in metadata["feature_descriptions"].items():
        st.markdown(f"- **{FEATURE_LABELS[name]}** (`{name}`) — {text}")
    st.markdown(
        f"**Label.** {metadata['historical']['positive_label']} Negative: {metadata['historical']['negative_label']}"
    )

# --- Model & validation ---------------------------------------------------
with tabs[1]:
    st.subheader("Selected model and preprocessing")
    st.markdown(
        f"- **Model:** {model['algorithm']} without rainfall (`{model['name']}`), {model['parameters']['n_estimators']} trees, "
        f"learning rate {model['parameters']['learning_rate']}, {model['parameters']['num_leaves']} leaves, seed {model['random_seed']}.\n"
        f"- **Features:** {', '.join(f'`{f}`' for f in model['features'])}; excluded: {', '.join(f'`{f}`' for f in model['excluded_features'])}.\n"
        "- **Preprocessing:** numeric features standardised and `land_cover_type` one-hot encoded, fitted inside each training fold "
        f"({model['transformed_feature_count']} encoded features).\n"
        f"- **Application predictions:** fitted once on all {model['training_observations']:,} development observations "
        f"({model['training_scope']}). Map scores are therefore in-sample.\n"
        f"- **Frozen cutoff:** {model['frozen_cutoff']:.2f} — the highest cutoff from 0.05 to 0.95 keeping area-fold recall at or above 0.85."
    )
    st.subheader("Model selection methodology")
    st.markdown(
        "Models were compared with **five geographic area-group folds** (grouped by `grid_id`, so a cell never appears in both "
        "training and validation) and **leave-one-storm-out** folds (each development storm held out in turn). Cutoffs were "
        "chosen on area-fold out-of-fold predictions and applied unchanged to the storm folds."
    )
    results = load_report_csv("results.csv")
    if results is not None:
        keep = ["logistic_regression_no_rain", "random_forest_no_rain", "xgboost_no_rain", "lightgbm_no_rain", "mlp_no_rain", "cnn_hybrid", "cnn_tile_only"]
        view = results[(results["fold"] == "mean") & results["model"].isin(keep)][
            ["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
        view["split"] = view["split"].map({"area_group": "Area groups", "leave_one_storm_out": "Leave one storm out"})
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})
        st.caption("Mean validation scores at the default 0.5 threshold, from `reports/results.csv`. LightGBM without rainfall had the "
                   "strongest ROC-AUC among final-model candidates on both designs. Development validation informed model selection.")
    st.markdown(
        f"At the frozen 0.50 cutoff: geographic recall **{ts['area_recall']:.2%}**, precision **{ts['area_precision']:.2%}**; "
        f"cross-storm recall **{ts['storm_recall']:.2%}**, precision **{ts['storm_precision']:.2%}** "
        "(observation-level, not grid-level)."
    )
    fig = figure_path("validation_model_comparison.png")
    if fig:
        st.image(str(fig), caption="Saved validation comparison (reports/figures/validation_model_comparison.png).", width="stretch")

# --- Independent test ------------------------------------------------------
with tabs[2]:
    st.subheader("One-time Imelda independent test")
    final = load_report_csv("final_test_results.csv")
    imelda = load_storm_metrics().query("evaluation_protocol == 'independent_test'").iloc[0]
    k = st.columns(6)
    for col, (key, label) in zip(k, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                                     ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
        col.metric(label, f"{imelda[key]:.4f}")
    st.markdown(
        f"Tropical Storm Imelda (2019) was reserved until the model, features, and cutoffs were frozen, then evaluated **once** on "
        f"{metadata['dataset_counts']['independent_test_observations']:,} cleaned observations. `reports/final_test.lock` prevents a rerun, "
        "and FloodSense only reads the saved results."
    )
    if final is not None:
        view = final[final["role"] == "final"].query("cutoff_type == 'frozen'")[
            ["model", "cutoff", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})
        st.caption("Final models at their frozen cutoffs, from `reports/final_test_results.csv`.")
    st.info("This is a single-storm result. About 3,800 Imelda grid cells were also in training, so it tests a new storm at mostly familiar places.")
    fig = figure_path("final_test_model_comparison.png")
    if fig:
        st.image(str(fig), caption="Saved final-test comparison at frozen cutoffs.", width="stretch")

# --- Tuning ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Hyperparameter tuning")
    st.markdown(
        "The team evaluated **randomized LightGBM hyperparameter configurations** (20 of 2,916 combinations per search, optimising "
        "ROC-AUC) using **nested cross-validation**: 5 outer × 4 inner geographic folds and 3 outer × 2 inner storm folds — 528 fits in total. "
        "The tested procedure **did not improve** the original model's mean validation metrics: none of the 12 aggregate comparisons "
        "improved, so the original configuration was retained."
    )
    overall = load_report_csv("hyperparameter_tuning/overall_comparison.csv")
    if overall is not None:
        overall = overall.set_index(overall.columns[0])
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.tuning_comparison(overall, "area"), width="stretch", config={"displayModeBar": False})
        c2.plotly_chart(charts.tuning_comparison(overall, "storm"), width="stretch", config={"displayModeBar": False})
        st.dataframe(overall, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.4f") for c in overall.columns})
        st.caption("From `reports/hyperparameter_tuning/overall_comparison.csv`. Differences are small and were not tested for statistical significance.")
    st.caption("This does not show the original configuration is globally optimal — only that this particular search did not improve it.")

# --- Rainfall ---------------------------------------------------------------
with tabs[4]:
    st.subheader("Why rainfall is excluded")
    st.markdown(
        "Rainfall varied sharply between storms. Hurricane Harvey's sampled maximum was **20.818 inches**, versus **13.216 inches** "
        "for the highest maximum among the other training storms, and the cleaned Imelda file's median was **1.480 inches** versus "
        "**8.017 inches** in cleaned training. With rainfall included, LightGBM's Harvey recall at 0.5 was **0.123**; without it, "
        "**0.879**. These differences, and the inconsistent cross-storm behaviour with rainfall, motivated dropping it. They do not "
        "prove that rainfall caused the model differences."
    )
    st.warning("Because rainfall is excluded, FloodSense cannot describe what any specific — or future — storm will do.")
    fig = figure_path("harvey_score_vs_rain.png")
    if fig:
        st.image(str(fig), caption="Saved diagnostic: LightGBM Harvey out-of-fold scores by rainfall bin (training-only; not causal).", width="stretch")

# --- SHAP --------------------------------------------------------------------
with tabs[5]:
    st.subheader("Two separate SHAP analyses")
    st.markdown(
        f"1. **Original global SHAP analysis** — {shap_meta['original_global_encoded_features']} encoded features, fixed random sample of "
        f"{shap_meta['original_global_sample_size']:,} training rows (`{shap_meta['original_global_importance_source']}`).\n"
        f"2. **Application-specific grouped SHAP** — {shap_meta['grouped_feature_count']} original features for all "
        f"{shap_meta['local_explanation_count']:,} unique sampled locations. {shap_meta['land_cover_grouping']} "
        "Grouped importance was computed from the location-level values — not by adding up the encoded mean |SHAP| values."
    )
    st.caption(f"Units: {shap_meta['units']}. {shap_meta['positive_contribution']} {shap_meta['negative_contribution']} SHAP is not causal.")
    c1, c2 = st.columns(2)
    encoded = load_global_shap()
    with c1:
        st.plotly_chart(charts.importance_bar(encoded, "mean_abs_shap", "feature", "1 · Original: 8 encoded features (5,000-row sample)", "#103D80"),
                        width="stretch", config={"displayModeBar": False})
    grouped = pd.DataFrame(
        [(FEATURE_LABELS[k], v) for k, v in shap_meta["grouped_mean_absolute_importance"].items()], columns=["feature", "mean_abs_shap"])
    with c2:
        st.plotly_chart(charts.importance_bar(grouped, "mean_abs_shap", "feature", "2 · App: 5 grouped features (62,138 locations)", "#46B8E9"),
                        width="stretch", config={"displayModeBar": False})
    with st.expander("Importance values as tables"):
        t1, t2 = st.columns(2)
        t1.dataframe(encoded, hide_index=True, width="stretch")
        t2.dataframe(grouped, hide_index=True, width="stretch")
        st.caption("The encoded table also lists LightGBM split-count importance and rank, which only partly agree with SHAP.")

    st.markdown("#### Original research SHAP figures")
    figures = [
        ("shap_lightgbm_mean_abs.png", "Mean absolute SHAP by encoded feature (original 5,000-row training sample)."),
        ("shap_lightgbm_beeswarm.png", "Beeswarm: each dot is one sampled training row; colour shows the feature value, position its SHAP contribution."),
        ("shap_dependence_elevation_m.png", "Dependence: how elevation's contribution varies across observed elevations."),
        ("shap_dependence_slope_degrees.png", "Dependence: slope's contribution across observed slopes."),
        ("shap_dependence_distance_to_bayou_m.png", "Dependence: distance-to-bayou contribution across observed distances."),
        ("shap_waterfall_flooded_correct.png", "Waterfall: an in-sample example correctly scored as report-associated."),
        ("shap_waterfall_not_flooded_correct.png", "Waterfall: an in-sample example correctly scored as not report-associated."),
        ("shap_waterfall_wrong_prediction.png", "Waterfall: an in-sample example the model got wrong."),
    ]
    cols = st.columns(2)
    for index, (name, caption) in enumerate(figures):
        path = figure_path(name)
        with cols[index % 2]:
            if path:
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.caption(f"Figure `{name}` is unavailable.")
    st.caption("Waterfall examples are in-sample illustrations, not independent performance estimates.")

# --- Limitations ------------------------------------------------------------
with tabs[6]:
    st.subheader("Research limitations")
    render_limitations(metadata)
    st.markdown(
        "- Geographic cells are about 1.11 km north–south by 0.96 km east–west.\n"
        "- Planner rankings support further investigation, not official decisions.\n"
        "- Labels come from 311 reports, not measured flood depth or verified property-level flooding."
    )
    st.subheader("Official flood information")
    render_official_resources()

# --- Team -------------------------------------------------------------------
with tabs[7]:
    st.subheader("Team and project")
    st.markdown("\n".join(f"- {name}" for name in TEAM))
    st.markdown("COSC 6368 · University of Houston")
    st.caption("Team information as recorded in the project README. FloodSense is a research prototype, not an official product of any agency.")
