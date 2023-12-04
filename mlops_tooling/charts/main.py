import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_optimal_pr_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    no_skill = sum(y_true) / len(y_true)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold_index = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_threshold_index]

    # Create figure
    fig = go.Figure()

    # Plot Precision-Recall Curve
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall Curve")
    )

    # Plot No Skill Curve
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, 1.01, 0.01),
            y=[no_skill] * 101,
            line=dict(color="grey", dash="dash"),
            name="No skill",
        )
    )

    # Highlight optimal threshold point
    fig.add_trace(
        go.Scatter(
            x=[recall[optimal_threshold_index]],
            y=[precision[optimal_threshold_index]],
            mode="markers",
            marker=dict(color="red", symbol="circle", size=10),
            name="Optimal Threshold",
        )
    )

    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        title="Precision-Recall Curve with Optimal Threshold",
        showlegend=True,
        template="simple_white",
    )

    fig.show()

    return optimal_threshold


def find_optimal_auc_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    youden_j = tpr - fpr

    optimal_threshold_index = youden_j.argmax()
    optimal_threshold = thresholds[optimal_threshold_index]

    # Create figure
    fig = go.Figure()

    # Plot Precision-Recall Curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))

    # Plot No Skill Curve
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, 1.01, 0.01),
            y=np.arange(0, 1.01, 0.01),
            line=dict(color="grey", dash="dash"),
            name="No skill",
        )
    )

    # Highlight optimal threshold point
    fig.add_trace(
        go.Scatter(
            x=[fpr[optimal_threshold_index]],
            y=[tpr[optimal_threshold_index]],
            mode="markers",
            marker=dict(color="red", symbol="circle", size=10),
            name="Optimal Threshold",
        )
    )

    fig.update_layout(
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        title="ROC Curve with Optimal Threshold",
        showlegend=True,
        template="simple_white",
    )

    fig.show()

    return optimal_threshold


def create_area_chart(
    dataset: pd.DataFrame,
    title: str,
    x: str,
    y: str,
    height: int = 600,
    width: int = 1200,
):
    """
    Creates area chart for one variable.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataset[x],
            y=dataset[y],
            name=x,
            stackgroup="one",
            marker=dict(color="#46039f"),
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(
            title_text=x,
            type="linear",
            tickangle=0,
            ticklabelstep=1,
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
        ),
        template="simple_white",
    )

    return fig


def create_area_charts(
    dataset: pd.DataFrame,
    title: str,
    subtitles: str,
    x1: str,
    y1: str,
    x2: str,
    y2: str,
    height: int = 600,
    width: int = 1200,
):
    """
    Creates two area charts side by side.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=subtitles)

    fig.add_trace(
        go.Scatter(
            x=dataset[x1],
            y=dataset[y1],
            stackgroup="one",
            marker=dict(color="#46039f"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dataset[x2],
            y=dataset[y2],
            stackgroup="one",
            marker=dict(color="#46039f"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        width=width,
        height=height,
        showlegend=False,
        template="simple_white",
        title_text=title,
    )

    return fig


def create_overlapping_area_charts(
    dataset: pd.DataFrame,
    title: str,
    y_title: str,
    x: str,
    y1: str,
    y2: str,
    height: int = 600,
    width: int = 1200,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataset[x],
            y=dataset[y1],
            name=y1,
            stackgroup="one",
            marker=dict(color="#46039f"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataset[x],
            y=dataset[y2],
            name=y2,
            stackgroup="two",
            marker=dict(color="#fb9f3a", size=0),
        )
    )

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        yaxis_title=y_title,
        xaxis=dict(
            title_text=x,
            type="linear",
            tickangle=0,
            ticklabelstep=1,
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
        ),
        template="simple_white",
    )

    return fig
