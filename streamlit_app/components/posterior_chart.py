"""Plotly bar chart for the RiskLevel posterior distribution."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


# Color per risk level — green→yellow→orange→red.
_COLORS = {
    "Low":      "#10B981",
    "Moderate": "#F59E0B",
    "High":     "#F97316",
    "Critical": "#DC2626",
}


def render_posterior(
    posterior: dict[str, float],
    title: str = "Risk Level Posterior",
    height: int = 280,
) -> None:
    """Render an interactive horizontal bar chart of the posterior.

    Bars are colored by severity. Argmax is highlighted with bold text.
    """
    states = list(posterior.keys())
    probs = [posterior[s] for s in states]
    colors = [_COLORS.get(s, "#888") for s in states]

    argmax = max(posterior, key=posterior.get)

    fig = go.Figure(go.Bar(
        x=probs,
        y=states,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=40, t=40, b=10),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            showgrid=True,
            gridcolor="rgba(150,150,150,0.2)",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(states)),  # Critical at top
            tickfont=dict(size=14),
        ),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Most likely:** :{_streamlit_color(argmax)}[{argmax}]  "
        f"({posterior[argmax]:.1%})"
    )


def _streamlit_color(state: str) -> str:
    """Map a state to a Streamlit-supported color name for inline markdown."""
    return {
        "Low":      "green",
        "Moderate": "orange",
        "High":     "red",
        "Critical": "red",
    }.get(state, "gray")
