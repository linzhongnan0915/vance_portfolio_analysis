"""Dashboard chart helpers."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def bar_target_weights(df, title: str) -> None:
    """Bar chart for output/live/latest_target_weights.csv (display only)."""
    if df.empty or "target_weight" not in df.columns or "ticker" not in df.columns:
        return
    plot_df = df.sort_values("target_weight", ascending=True)
    fig = px.bar(
        plot_df,
        x="target_weight",
        y="ticker",
        orientation="h",
        title=title,
        text="target_weight",
    )
    fig.update_traces(texttemplate="%{x:.1%}", textposition="outside")
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Target weight", yaxis_title="")
    st.plotly_chart(fig, width="stretch")


def pie_weights(weights: dict, title: str) -> None:
    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        title=title,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width="stretch")
