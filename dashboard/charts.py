"""Dashboard chart helpers."""

from __future__ import annotations

import plotly.express as px
import streamlit as st


def pie_weights(weights: dict, title: str) -> None:
    fig = px.pie(
        names=list(weights.keys()),
        values=list(weights.values()),
        title=title,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width="stretch")
