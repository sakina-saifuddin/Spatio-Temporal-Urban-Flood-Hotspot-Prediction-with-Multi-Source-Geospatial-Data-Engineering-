"""Plotly charts with FloodSense styling. Each chart is paired in the pages
with a data table or text alternative for accessibility."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils.grid_utils import CLASS_COLORS, CLASS_LABELS, FEATURE_LABELS, FEATURE_SHORT_LABELS

POSITIVE = "#D9620B"  # orange: pushes the raw model output up
NEGATIVE = "#2865DF"  # blue: pushes it down
INK = "#0F1E36"
FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def _layout(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="Poppins, " + FONT, size=16, color=INK)) if title else None,
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=24, t=56 if title else 20, b=64),
        height=height,
    )
    # automargin keeps long feature names and axis titles from being clipped.
    fig.update_xaxes(automargin=True, title_standoff=10)
    fig.update_yaxes(automargin=True)
    return fig


def shap_frame(row: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "feature": list(FEATURE_LABELS),
            "label": [FEATURE_LABELS[f] for f in FEATURE_LABELS],
            "contribution": [float(row[f"shap_{f}"]) for f in FEATURE_LABELS],
        }
    )
    data["direction"] = data["contribution"].map(lambda v: "Raises score" if v > 0 else ("Lowers score" if v < 0 else "No effect"))
    return data.sort_values("contribution", key=lambda s: s.abs())


def shap_bar(row: pd.Series, title: str | None = None) -> go.Figure:
    data = shap_frame(row)
    colors = [POSITIVE if v > 0 else NEGATIVE for v in data["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=data["contribution"],
            y=data["label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f} ({'▲ up' if v > 0 else '▼ down'})" for v in data["contribution"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>",
        )
    )
    span = max(0.5, data["contribution"].abs().max() * 1.75)
    fig.add_vline(x=0, line_color=INK, line_width=1.5)
    fig.update_xaxes(title="Mean SHAP contribution (raw model log-odds)", range=[-span, span], zeroline=False, gridcolor="#E2E8F0")
    fig.update_yaxes(title=None, ticklabelstandoff=6)
    return _layout(fig, title, height=330)


def shap_compare(row_a: pd.Series, row_b: pd.Series, name_a: str, name_b: str) -> go.Figure:
    labels = [FEATURE_SHORT_LABELS[f] for f in FEATURE_LABELS]
    fig = go.Figure()
    for row, name, color in ((row_a, name_a, "#103D80"), (row_b, name_b, "#46B8E9")):
        values = [float(row[f"shap_{f}"]) for f in FEATURE_LABELS]
        fig.add_bar(x=values, y=labels, orientation="h", name=name, marker_color=color,
                    text=[f"{v:+.2f}" for v in values], textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_color=INK, line_width=1.5)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_xaxes(title="Mean SHAP contribution (raw log-odds)", gridcolor="#E2E8F0")
    return _layout(fig, "SHAP contributions side by side", height=380)


def confusion_matrix(metrics_row: pd.Series) -> go.Figure:
    tp, tn, fp, fn = (int(metrics_row[k]) for k in ("tp", "tn", "fp", "fn"))
    z = [[tn, fp], [fn, tp]]
    total = tp + tn + fp + fn
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{CLASS_LABELS[labels[r][c]]}<br>{z[r][c]:,}<br>({z[r][c] / total:.1%})" for c in range(2)] for r in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted: below 0.50", "Predicted: at or above 0.50"],
            y=["Recorded: no report", "Recorded: report"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#EFF6FF"], [1, "#2865DF"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>%{z:,} observations<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Confusion matrix at the frozen 0.50 cutoff", height=340)


def storm_comparison(metrics: pd.DataFrame, selected: str | None = None) -> go.Figure:
    names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    pretty = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC (AP)"]
    palette = ["#103D80", "#2865DF", "#46B8E9", "#8C3F08"]
    fig = go.Figure()
    for index, row in metrics.reset_index(drop=True).iterrows():
        label = f"{row['storm_name']} ({int(row['year'])})" + (" — independent test" if row["evaluation_protocol"] == "independent_test" else "")
        fig.add_bar(
            x=pretty, y=[row[n] for n in names], name=label,
            marker_color=palette[index % len(palette)],
            marker_line=dict(width=3 if row["storm_name"] == selected else 0, color="#F8D04B"),
            hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>",
        )
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.3))
    fig.update_yaxes(range=[0.5, 1.0], title="Score", gridcolor="#E2E8F0")
    return _layout(fig, "Cross-storm performance (frozen 0.50 cutoff)", height=420)


def classification_breakdown(breakdown: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[CLASS_LABELS[c] for c in breakdown["classification"]],
            y=breakdown["count"],
            marker_color=[CLASS_COLORS[c] for c in breakdown["classification"]],
            text=[f"{n} ({s:.0%})" for n, s in zip(breakdown["count"], breakdown["share"])],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(title="Observations", gridcolor="#E2E8F0")
    return _layout(fig, "Classifications in the selected cell", height=300)


def importance_bar(frame: pd.DataFrame, value_col: str, label_col: str, title: str, color: str = "#2865DF") -> go.Figure:
    data = frame.sort_values(value_col)
    fig = go.Figure(
        go.Bar(x=data[value_col], y=data[label_col], orientation="h", marker_color=color,
               text=[f"{v:.3f}" for v in data[value_col]], textposition="outside", cliponaxis=False)
    )
    fig.update_xaxes(title="Mean |SHAP| (raw log-odds)", gridcolor="#E2E8F0")
    return _layout(fig, title, height=60 + 38 * len(data))


def score_histogram(scores: pd.Series) -> go.Figure:
    # No cutoff line: the 0.50 cutoff is observation-level and was never
    # validated as a threshold for grid-cell mean scores.
    fig = go.Figure(go.Histogram(x=scores, nbinsx=40, marker_color="#46B8E9"))
    fig.update_xaxes(title="Mean cell susceptibility score", range=[0, 1])
    fig.update_yaxes(title="Cells", gridcolor="#E2E8F0")
    return _layout(fig, "Score distribution of filtered cells", height=280)


def tuning_comparison(frame: pd.DataFrame, split: str) -> go.Figure:
    pretty = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    metrics = [pretty.get(m, m) for m in frame.index]
    fig = go.Figure()
    fig.add_bar(x=metrics, y=frame[f"baseline_{split}"], name="Original configuration", marker_color="#103D80")
    fig.add_bar(x=metrics, y=frame[f"tuned_{split}"], name="Tuned (nested CV)", marker_color="#46B8E9")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(range=[0.75, 0.92], gridcolor="#E2E8F0", title="Mean outer-fold score")
    title = "Geographic CV" if split == "area" else "Storm CV"
    return _layout(fig, title, height=340)
