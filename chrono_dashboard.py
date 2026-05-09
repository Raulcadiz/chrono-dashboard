"""
Chrono Dashboard — Universal time-series event correlation explorer.
Built on chrono-correlator 1.2.0 (Apache 2.0) by Raúl Gallardo.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chrono_correlator import (
    PROMPT_TEMPLATES,
    AlertReport,
    CorrelationResult,
    Event,
    Metric,
    SignificanceConfig,
    evaluate,
    export_html,
    export_markdown,
    find_best_lag,
    generate_events,
    generate_metric,
    inject_pattern,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chrono Dashboard",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Visual constants
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_COLOR = {"strong": "#ef4444", "moderate": "#f97316", "weak": "#eab308", "none": "#6b7280"}
LEVEL_COLOR  = {"red": "#ef4444", "yellow": "#f97316", "green": "#22c55e"}
LEVEL_LABEL  = {"red": "🔴 Alerta alta", "yellow": "🟡 Señal moderada", "green": "🟢 Sin señal"}

# ──────────────────────────────────────────────────────────────────────────────
# Domain presets
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS: dict[str, dict] = {
    "Salud Personal": {
        "icon": "🫀",
        "desc": "HRV, sueño, síntomas y eventos de salud personal",
        "metric_name": "HRV (ms)",
        "event_label": "Síntoma / Evento",
        "lookback_hours": 72,
        "baseline_days": 28,
        "direction": "decrease",
        "baseline_strategy": "rolling",
        "prompt_template": "default",
        "lang": "es",
        "gen": {"base_value": 55.0, "noise_std": 3.5, "days": 60, "circadian_amplitude": 6.0},
        "n_events": 8,
        "pattern_dir": "decrease",
        "unit": "ms",
    },
    "Salud Pública": {
        "icon": "🏥",
        "desc": "Listas de espera, indicadores sanitarios y políticas públicas",
        "metric_name": "Pacientes en lista de espera",
        "event_label": "Cambio de política",
        "lookback_hours": 168,
        "baseline_days": 60,
        "direction": "increase",
        "baseline_strategy": "rolling",
        "prompt_template": "default",
        "lang": "es",
        "gen": {"base_value": 2500.0, "noise_std": 80.0, "days": 120, "circadian_amplitude": 0.0},
        "n_events": 5,
        "pattern_dir": "increase",
        "unit": "pacientes",
    },
    "Finanzas y Trading": {
        "icon": "📈",
        "desc": "Precios, volumen y eventos de mercado financiero",
        "metric_name": "Precio de cierre",
        "event_label": "Evento de mercado",
        "lookback_hours": 48,
        "baseline_days": 30,
        "direction": "two-sided",
        "baseline_strategy": "same_weekday",
        "prompt_template": "finance",
        "lang": "en",
        "gen": {"base_value": 100.0, "noise_std": 2.5, "days": 60, "circadian_amplitude": 1.0},
        "n_events": 6,
        "pattern_dir": "decrease",
        "unit": "USD",
    },
    "Negocios y Ventas": {
        "icon": "💼",
        "desc": "KPIs de negocio, campañas y lanzamientos de producto",
        "metric_name": "Ingresos diarios (€)",
        "event_label": "Campaña / Lanzamiento",
        "lookback_hours": 168,
        "baseline_days": 45,
        "direction": "increase",
        "baseline_strategy": "same_weekday",
        "prompt_template": "default",
        "lang": "es",
        "gen": {"base_value": 5000.0, "noise_std": 500.0, "days": 90, "circadian_amplitude": 0.0},
        "n_events": 6,
        "pattern_dir": "increase",
        "unit": "€",
    },
    "Infraestructura y DevOps": {
        "icon": "🖥️",
        "desc": "Latencia, errores y eventos de despliegue / incidentes",
        "metric_name": "Latencia p95 (ms)",
        "event_label": "Deployment / Incidente",
        "lookback_hours": 24,
        "baseline_days": 14,
        "direction": "increase",
        "baseline_strategy": "same_hour",
        "prompt_template": "it",
        "lang": "en",
        "gen": {"base_value": 85.0, "noise_std": 8.0, "days": 30, "circadian_amplitude": 10.0},
        "n_events": 8,
        "pattern_dir": "increase",
        "unit": "ms",
    },
    "Política y Ciencias Sociales": {
        "icon": "🏛️",
        "desc": "Indicadores sociales y eventos político-institucionales",
        "metric_name": "Índice de aprobación (%)",
        "event_label": "Evento político",
        "lookback_hours": 336,
        "baseline_days": 90,
        "direction": "two-sided",
        "baseline_strategy": "rolling",
        "prompt_template": "science",
        "lang": "en",
        "gen": {"base_value": 45.0, "noise_std": 2.0, "days": 180, "circadian_amplitude": 0.0},
        "n_events": 6,
        "pattern_dir": "decrease",
        "unit": "%",
    },
    "Ciencia e Investigación": {
        "icon": "🔬",
        "desc": "Mediciones experimentales e intervenciones de laboratorio",
        "metric_name": "Respuesta experimental",
        "event_label": "Intervención",
        "lookback_hours": 96,
        "baseline_days": 30,
        "direction": "two-sided",
        "baseline_strategy": "rolling",
        "prompt_template": "science",
        "lang": "en",
        "gen": {"base_value": 100.0, "noise_std": 5.0, "days": 60, "circadian_amplitude": 0.0},
        "n_events": 7,
        "pattern_dir": "decrease",
        "unit": "u.a.",
    },
    "Energía y Medio Ambiente": {
        "icon": "⚡",
        "desc": "Consumo energético, producción y eventos climáticos",
        "metric_name": "Consumo eléctrico (MW)",
        "event_label": "Evento energético",
        "lookback_hours": 48,
        "baseline_days": 28,
        "direction": "increase",
        "baseline_strategy": "same_hour",
        "prompt_template": "default",
        "lang": "es",
        "gen": {"base_value": 4000.0, "noise_std": 200.0, "days": 60, "circadian_amplitude": 600.0},
        "n_events": 6,
        "pattern_dir": "increase",
        "unit": "MW",
    },
    "Marketing Digital": {
        "icon": "📣",
        "desc": "Tráfico web, conversiones y campañas de marketing",
        "metric_name": "Sesiones diarias",
        "event_label": "Campaña / Publicación",
        "lookback_hours": 168,
        "baseline_days": 30,
        "direction": "increase",
        "baseline_strategy": "same_weekday",
        "prompt_template": "default",
        "lang": "es",
        "gen": {"base_value": 10000.0, "noise_std": 800.0, "days": 60, "circadian_amplitude": 0.0},
        "n_events": 7,
        "pattern_dir": "increase",
        "unit": "sesiones",
    },
    "Personalizado": {
        "icon": "🔧",
        "desc": "Sube tus propios datos y configura el análisis manualmente",
        "metric_name": "Mi métrica",
        "event_label": "Mi evento",
        "lookback_hours": 48,
        "baseline_days": 28,
        "direction": "two-sided",
        "baseline_strategy": "rolling",
        "prompt_template": "default",
        "lang": "es",
        "gen": None,
        "n_events": 5,
        "pattern_dir": "decrease",
        "unit": "u",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _boost_pattern(
    metric: Metric,
    events: list[Event],
    lookback_hours: int = 72,
    boost_fraction: float = 0.30,
) -> Metric:
    """Mirror of inject_pattern but raises values before events (for 'increase' domains)."""
    ts = np.array([t.timestamp() for t in metric.timestamps])
    values = np.array(metric.values, dtype=float)
    for event in events:
        t_evt = event.timestamp.timestamp()
        t_start = t_evt - lookback_hours * 3600
        mask = (ts >= t_start) & (ts < t_evt)
        if mask.any():
            progress = (ts[mask] - t_start) / (t_evt - t_start)
            boost = boost_fraction * np.minimum(progress * 2.0, 1.0)
            values[mask] *= 1.0 + boost
    return Metric(name=metric.name, timestamps=metric.timestamps, values=values.tolist())


@st.cache_data(show_spinner=False)
def load_preset_data(domain_key: str) -> tuple[Metric, list[Event]]:
    cfg = DOMAINS[domain_key]
    gen = cfg["gen"]
    if gen is None:
        raise ValueError("El dominio 'Personalizado' no tiene datos de ejemplo.")

    start = datetime(2024, 1, 1)
    lookback = cfg["lookback_hours"]
    min_gap = max(3, lookback // 48)

    metric = generate_metric(
        name=cfg["metric_name"],
        start=start,
        days=gen["days"],
        base_value=gen["base_value"],
        noise_std=gen["noise_std"],
        circadian_amplitude=gen.get("circadian_amplitude", 6.0),
        seed=42,
    )
    events = generate_events(
        start=start,
        days=gen["days"],
        n_events=cfg["n_events"],
        min_gap_days=min_gap,
        label=cfg["event_label"],
        seed=42,
    )

    if cfg["pattern_dir"] == "decrease":
        metric = inject_pattern(metric, events, lookback_hours=lookback, drop_fraction=0.30)
    else:
        metric = _boost_pattern(metric, events, lookback_hours=lookback, boost_fraction=0.30)

    return metric, events


def parse_uploaded_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                file.seek(0)
                return pd.read_csv(file, encoding=enc)
            except Exception:
                continue
        raise ValueError("No se pudo leer el CSV con ninguna codificación estándar.")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        raise ValueError(f"Formato no soportado: '{file.name}'. Usa CSV o Excel (.xlsx/.xls).")


def detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    date_kw  = ("date", "fecha", "time", "timestamp", "dt", "hora", "day", "dia")
    value_kw = ("value", "valor", "metric", "metrica", "cantidad", "count", "measure", "v")
    event_kw = ("event", "evento", "label", "etiqueta", "type", "tipo", "category", "cat")

    date_col = value_col = event_col = None
    for col in df.columns:
        cl = col.lower().strip()
        if date_col is None and any(k in cl for k in date_kw):
            date_col = col
        elif value_col is None and any(k in cl for k in value_kw):
            value_col = col
        elif event_col is None and any(k in cl for k in event_kw):
            event_col = col

    # Fallbacks
    if date_col is None and len(df.columns) >= 1:
        date_col = df.columns[0]
    if value_col is None:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        if nums:
            value_col = nums[0]

    return {"date": date_col, "value": value_col, "event": event_col}


def build_metric_and_events(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    event_col: Optional[str],
    metric_name: str,
    event_label: str,
) -> tuple[Metric, list[Event]]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, utc=False)
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

    metric = Metric(
        name=metric_name,
        timestamps=[ts.to_pydatetime() for ts in df[date_col]],
        values=df[value_col].astype(float).tolist(),
    )

    events: list[Event] = []
    if event_col and event_col in df.columns:
        mask = df[event_col].notna() & (df[event_col].astype(str).str.strip() != "")
        for _, row in df[mask].iterrows():
            events.append(Event(
                timestamp=row[date_col].to_pydatetime(),
                label=str(row[event_col]).strip() or event_label,
            ))

    return metric, events

# ──────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(
    events: list[Event],
    metrics: list[Metric],
    lookback_hours: int,
    baseline_days: int,
    direction: str,
    baseline_strategy: str,
    correction: Optional[str],
    bootstrap_ci: bool,
    config: SignificanceConfig,
) -> AlertReport:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(
            events=events,
            metrics=metrics,
            lookback_hours=lookback_hours,
            baseline_days=baseline_days,
            direction=direction,
            baseline_strategy=baseline_strategy,
            correction=correction,
            bootstrap_ci=bootstrap_ci,
            config=config,
        )


def run_lag_sweep(
    events: list[Event],
    metric: Metric,
    max_lag: int,
    step: int,
    lookback_hours: int,
    baseline_days: int,
    direction: str,
    baseline_strategy: str,
    config: SignificanceConfig,
) -> dict[int, CorrelationResult]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return find_best_lag(
            events=events,
            metric=metric,
            lag_range=range(0, max_lag + step, step),
            lookback_hours=lookback_hours,
            baseline_days=baseline_days,
            direction=direction,
            baseline_strategy=baseline_strategy,
            config=config,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    metric: Metric,
    events: list[Event],
    report: Optional[AlertReport],
    lookback_hours: int,
) -> go.Figure:
    fig = go.Figure()

    # Raw series
    fig.add_trace(go.Scatter(
        x=metric.timestamps,
        y=metric.values,
        mode="lines",
        name=metric.name,
        line=dict(color="#3b82f6", width=1.2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.2f}</b><extra>" + metric.name + "</extra>",
    ))

    # Rolling mean overlay
    n = max(1, len(metric.values) // 20)
    s = pd.Series(metric.values, index=metric.timestamps)
    roll = s.rolling(n, center=True).mean()
    fig.add_trace(go.Scatter(
        x=roll.index.tolist(),
        y=roll.values.tolist(),
        mode="lines",
        name="Media móvil",
        line=dict(color="#93c5fd", width=2, dash="dash"),
        opacity=0.8,
    ))

    # Pre-event shading for significant results
    if report:
        for r in report.results:
            if r.significant:
                for evt in events:
                    t1 = (evt.timestamp - timedelta(hours=lookback_hours)).isoformat()
                    t2 = evt.timestamp.isoformat()
                    fig.add_vrect(
                        x0=t1, x1=t2,
                        fillcolor="rgba(239,68,68,0.07)",
                        line_width=0,
                    )

    # Event vertical lines — pass ISO string, not datetime (plotly sum() bug in Py3.13)
    for evt in events:
        fig.add_vline(
            x=evt.timestamp.isoformat(),
            line_width=1.5,
            line_dash="dot",
            line_color="#f97316",
            annotation_text=evt.label[:16],
            annotation_position="top",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=dict(text=f"Serie temporal · {metric.name}", font_size=14),
        xaxis_title="Fecha",
        yaxis_title=metric.name,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=430,
        margin=dict(l=55, r=20, t=65, b=50),
    )
    return fig


def fig_before_after(
    metric: Metric,
    events: list[Event],
    lookback_hours: int,
    baseline_days: int,
) -> go.Figure:
    ts  = np.array([t.timestamp() for t in metric.timestamps])
    vals = np.array(metric.values)

    pre_vals, base_vals = [], []
    for evt in events:
        t_evt = evt.timestamp.timestamp()
        t_pre_end   = t_evt
        t_pre_start = t_pre_end - lookback_hours * 3600
        t_base_start = t_evt - baseline_days * 86400

        pre_vals.extend(vals[(ts >= t_pre_start) & (ts < t_pre_end)].tolist())
        base_vals.extend(vals[(ts >= t_base_start) & (ts < t_pre_start)].tolist())

    fig = go.Figure()
    if base_vals:
        fig.add_trace(go.Box(y=base_vals, name="Baseline", marker_color="#6b7280",
                             boxmean="sd", hoverinfo="y"))
    if pre_vals:
        fig.add_trace(go.Box(y=pre_vals, name="Pre-evento", marker_color="#ef4444",
                             boxmean="sd", hoverinfo="y"))

    fig.update_layout(
        title=dict(text="Distribución: Baseline vs. Pre-evento", font_size=14),
        yaxis_title=metric.name,
        height=380,
        margin=dict(l=55, r=20, t=65, b=50),
        showlegend=True,
    )
    return fig


def fig_lag_sweep(lag_results: dict[int, CorrelationResult]) -> go.Figure:
    if not lag_results:
        return go.Figure()

    lags      = sorted(lag_results.keys())
    strengths = [lag_results[l].association_strength for l in lags]
    p_vals    = [lag_results[l].p_value for l in lags]
    sig       = [lag_results[l].significant for l in lags]
    colors    = [SIGNAL_COLOR[lag_results[l].signal_strength] for l in lags]

    fig = go.Figure(go.Bar(
        x=lags,
        y=strengths,
        marker_color=colors,
        customdata=list(zip(p_vals, [lag_results[l].signal_strength for l in lags])),
        hovertemplate=(
            "Lag: %{x}h<br>"
            "Asociación: %{y:.3f}<br>"
            "p-valor: %{customdata[0]:.4f}<br>"
            "Señal: %{customdata[1]}<extra></extra>"
        ),
    ))

    best_lag = max(lag_results, key=lambda k: lag_results[k].association_strength)
    fig.add_vline(
        x=best_lag,
        line_color="#22c55e",
        line_dash="dash",
        annotation_text=f"Mejor lag: {best_lag}h",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(text="Barrido de lag — Fuerza de asociación por ventana", font_size=14),
        xaxis_title="Lag (horas)",
        yaxis_title="Association strength (0–1)",
        height=360,
        margin=dict(l=55, r=20, t=65, b=50),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────────────────

def build_results_df(report: AlertReport) -> pd.DataFrame:
    rows = []
    for r in report.results:
        p_show = r.adjusted_p_value if r.adjusted_p_value is not None else r.p_value
        ci_str = (
            f"[{r.effect_ci[0]:+.3f}, {r.effect_ci[1]:+.3f}]"
            if r.effect_ci else "—"
        )
        rows.append({
            "Métrica":            r.metric_name,
            "p-valor":            f"{r.p_value:.4f}",
            "p-valor adj.":       f"{p_show:.4f}",
            "Efecto":             f"{r.effect_size:+.3f}",
            "IC 95%":             ci_str,
            "Consistencia":       f"{r.consistency:.0%}",
            "Señal":              r.signal_strength.capitalize(),
            "Asociación":         f"{r.association_strength:.3f}",
            "✓":                  "✅" if r.significant else "❌",
            "Baseline (med.)":    f"{r.baseline_median:.3f}",
            "Pre-evento (med.)":  f"{r.pre_event_median:.3f}",
        })
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Template-based narrative (no LLM required)
# ──────────────────────────────────────────────────────────────────────────────

def build_template_narrative(report: AlertReport, domain_cfg: dict, tone: str) -> str:
    lang = domain_cfg.get("lang", "es")
    es = lang == "es"
    lines: list[str] = []

    # Header
    lines.append(f"**Nivel de alerta: {LEVEL_LABEL[report.level]}**")
    lines.append(f"Señales activas: {report.active_signals} / {report.total_signals}")
    lines.append("")

    significant = [r for r in report.results if r.significant]

    if not significant:
        if es:
            lines.append(
                "No se encontraron asociaciones estadísticamente significativas con los parámetros actuales. "
                "Prueba a ajustar la ventana pre-evento, los días de baseline o la dirección de hipótesis."
            )
        else:
            lines.append(
                "No statistically significant associations were found with the current parameters. "
                "Try adjusting the pre-event window, baseline days, or hypothesis direction."
            )
        return "\n\n".join(lines)

    for r in significant:
        pct = (
            abs((r.pre_event_median - r.baseline_median) / r.baseline_median * 100)
            if r.baseline_median != 0 else 0.0
        )
        if es:
            direction_word = "incremento" if r.pre_event_median > r.baseline_median else "disminución"
        else:
            direction_word = "increase" if r.pre_event_median > r.baseline_median else "decrease"

        if tone == "Técnico":
            if es:
                line = (
                    f"**{r.metric_name}** — Mann-Whitney U: p={r.p_value:.4f}, "
                    f"rank-biserial={r.effect_size:+.3f}, consistencia={r.consistency:.2f}, "
                    f"asociación={r.association_strength:.3f} [{r.signal_strength}]. "
                    f"Mediana baseline: {r.baseline_median:.3f} → pre-evento: {r.pre_event_median:.3f}."
                )
            else:
                line = (
                    f"**{r.metric_name}** — Mann-Whitney U: p={r.p_value:.4f}, "
                    f"rank-biserial={r.effect_size:+.3f}, consistency={r.consistency:.2f}, "
                    f"association={r.association_strength:.3f} [{r.signal_strength}]. "
                    f"Baseline median: {r.baseline_median:.3f} → pre-event: {r.pre_event_median:.3f}."
                )
        elif tone == "Divulgativo":
            if es:
                line = (
                    f"**{r.metric_name}**: Antes de los eventos se observó un {direction_word} "
                    f"del {pct:.1f}% respecto al período de referencia "
                    f"(de {r.baseline_median:.2f} a {r.pre_event_median:.2f} {domain_cfg['unit']}). "
                    f"Este patrón apareció de forma consistente en el {r.consistency:.0%} de los "
                    f"eventos analizados (p={r.p_value:.4f})."
                )
            else:
                line = (
                    f"**{r.metric_name}**: A {pct:.1f}% {direction_word} was observed before events "
                    f"compared to the reference period "
                    f"(from {r.baseline_median:.2f} to {r.pre_event_median:.2f} {domain_cfg['unit']}). "
                    f"This pattern appeared consistently in {r.consistency:.0%} of analyzed events "
                    f"(p={r.p_value:.4f})."
                )
        elif tone == "Profesional":
            if es:
                line = (
                    f"**{r.metric_name}**: Se identifica una asociación temporal de intensidad "
                    f"*{r.signal_strength}* entre los eventos registrados y la variable analizada. "
                    f"La mediana en el período pre-evento ({r.pre_event_median:.2f}) difiere del "
                    f"baseline ({r.baseline_median:.2f}) con significancia estadística "
                    f"p={r.p_value:.4f} y consistencia del {r.consistency:.0%}."
                )
            else:
                line = (
                    f"**{r.metric_name}**: A *{r.signal_strength}* temporal association was identified "
                    f"between the recorded events and the analyzed variable. "
                    f"The pre-event median ({r.pre_event_median:.2f}) differs from baseline "
                    f"({r.baseline_median:.2f}) with statistical significance p={r.p_value:.4f} "
                    f"and consistency of {r.consistency:.0%}."
                )
        else:  # Neutral
            if es:
                line = (
                    f"**{r.metric_name}**: Señal *{r.signal_strength}* detectada. "
                    f"p={r.p_value:.4f} · efecto={r.effect_size:+.3f} · "
                    f"consistencia={r.consistency:.0%} · asociación={r.association_strength:.3f}."
                )
            else:
                line = (
                    f"**{r.metric_name}**: *{r.signal_strength.capitalize()}* signal detected. "
                    f"p={r.p_value:.4f} · effect={r.effect_size:+.3f} · "
                    f"consistency={r.consistency:.0%} · association={r.association_strength:.3f}."
                )
        lines.append(line)

    lines.append("")
    lines.append(
        "_Nota: Las asociaciones son estadísticas (Mann-Whitney U). "
        "No implican causalidad ni permiten diagnósticos._"
        if es else
        "_Note: Associations are statistical (Mann-Whitney U). "
        "They do not imply causality or support clinical diagnosis._"
    )

    return "\n\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# LLM narration
# ──────────────────────────────────────────────────────────────────────────────

def try_llm_narration(
    report: AlertReport,
    provider: str,
    api_key: str,
    prompt_template: str,
) -> tuple[AlertReport, Optional[str]]:
    if report.level == "green":
        return report, None
    if not any(r.significant for r in report.results):
        return report, "No hay señales significativas que narrar."

    key = api_key.strip() or None

    try:
        if provider == "anthropic":
            from chrono_correlator import AnthropicNarrator
            narrator = AnthropicNarrator(api_key=key, prompt_template=prompt_template or None)
        elif provider == "groq":
            from chrono_correlator import GroqNarrator
            narrator = GroqNarrator(api_key=key, prompt_template=prompt_template or None)
        elif provider == "ollama":
            from chrono_correlator import OllamaNarrator
            narrator = OllamaNarrator(prompt_template=prompt_template or None)
        else:
            return report, f"Proveedor desconocido: {provider}"

        return narrator.narrate(report), None

    except ImportError as exc:
        return report, f"Dependencia no instalada: {exc}. Instala con `pip install chrono-correlator[{provider}]`."
    except RuntimeError as exc:
        return report, str(exc)
    except Exception as exc:
        return report, f"Error al generar narrativa con IA: {exc}"

# ──────────────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────────────

def export_to_json(report: AlertReport, metric: Metric, events: list[Event]) -> str:
    data = {
        "generated_at":   datetime.now().isoformat(),
        "alert_level":    report.level,
        "active_signals": report.active_signals,
        "total_signals":  report.total_signals,
        "metric": {
            "name":       metric.name,
            "n_points":   len(metric.timestamps),
            "date_range": [metric.timestamps[0].isoformat(), metric.timestamps[-1].isoformat()],
        },
        "events": [{"timestamp": e.timestamp.isoformat(), "label": e.label} for e in events],
        "results": [
            {
                "metric_name":        r.metric_name,
                "p_value":            r.p_value,
                "adjusted_p_value":   r.adjusted_p_value,
                "effect_size":        r.effect_size,
                "effect_ci":          list(r.effect_ci) if r.effect_ci else None,
                "baseline_median":    r.baseline_median,
                "pre_event_median":   r.pre_event_median,
                "consistency":        r.consistency,
                "signal_strength":    r.signal_strength,
                "association_strength": r.association_strength,
                "significant":        r.significant,
                "narrative":          r.narrative,
            }
            for r in report.results
        ],
        "narrative": report.narrative,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_to_html_bytes(report: AlertReport) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    export_html(report, path)
    with open(path, "rb") as f:
        content = f.read()
    os.unlink(path)
    return content


def export_to_markdown_str(report: AlertReport) -> str:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        path = f.name
    export_markdown(report, path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    os.unlink(path)
    return content

# ──────────────────────────────────────────────────────────────────────────────
# Tab renderers
# ──────────────────────────────────────────────────────────────────────────────

def render_tab_datos(metric: Metric, events: list[Event]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    n = len(metric.timestamps)
    span_days = (metric.timestamps[-1] - metric.timestamps[0]).days if n > 1 else 0
    vals = np.array(metric.values)

    c1.metric("Puntos de datos", f"{n:,}")
    c2.metric("Período", f"{span_days} días")
    c3.metric("Eventos", len(events))
    c4.metric("Rango de valores",
              f"{vals.min():.2f} – {vals.max():.2f}",
              delta=f"μ={vals.mean():.2f}")

    st.divider()

    col_m, col_e = st.columns([2, 1])

    with col_m:
        st.markdown("#### Vista previa de la métrica")
        df_prev = pd.DataFrame({
            "Fecha":  metric.timestamps,
            "Valor":  metric.values,
        })
        st.dataframe(df_prev.tail(50), use_container_width=True, height=280)

    with col_e:
        st.markdown("#### Eventos registrados")
        if events:
            df_ev = pd.DataFrame([
                {"Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M"), "Etiqueta": e.label}
                for e in events
            ])
            st.dataframe(df_ev, use_container_width=True, height=280)
        else:
            st.info("No hay eventos registrados.")

    st.divider()
    st.markdown("#### Estadísticas descriptivas")
    desc = pd.Series(metric.values, name=metric.name).describe()
    st.dataframe(
        desc.to_frame().T.style.format("{:.3f}"),
        use_container_width=True,
    )


def render_tab_viz(
    metric: Metric,
    events: list[Event],
    report: AlertReport,
    lookback_hours: int,
    baseline_days: int,
) -> None:
    st.plotly_chart(
        fig_time_series(metric, events, report, lookback_hours),
        use_container_width=True,
    )

    col_box, col_gap = st.columns([1, 1])
    with col_box:
        st.plotly_chart(
            fig_before_after(metric, events, lookback_hours, baseline_days),
            use_container_width=True,
        )
    with col_gap:
        st.markdown("#### Sobre este gráfico")
        st.markdown(
            "Compara la distribución de la métrica durante el **período de baseline** "
            "(comportamiento histórico normal) frente al **período pre-evento** "
            "(ventana inmediatamente anterior a cada evento).\n\n"
            "Si las distribuciones son claramente diferentes, es probable que el "
            "análisis estadístico detecte señales significativas."
        )
        if report and report.results:
            r = report.results[0]
            st.markdown(f"""
| Estadístico | Valor |
|---|---|
| Baseline mediana | `{r.baseline_median:.3f}` |
| Pre-evento mediana | `{r.pre_event_median:.3f}` |
| Efecto (rank-biserial) | `{r.effect_size:+.3f}` |
| Consistencia | `{r.consistency:.0%}` |
""")


def render_tab_results(
    report: AlertReport,
    metric: Metric,
    events: list[Event],
    lookback_hours: int,
    baseline_days: int,
    direction: str,
    baseline_strategy: str,
    config: SignificanceConfig,
) -> None:
    st.markdown("#### Tabla de resultados estadísticos")
    df_res = build_results_df(report)

    def _color_sig(val: str) -> str:
        return "color: #22c55e; font-weight:600" if val == "✅" else "color: #9ca3af"

    def _color_signal(val: str) -> str:
        c = {"Strong": "#ef4444", "Moderate": "#f97316", "Weak": "#eab308"}.get(val, "#6b7280")
        return f"color: {c}; font-weight:600"

    styled = (
        df_res.style
        .applymap(_color_sig, subset=["✓"])
        .applymap(_color_signal, subset=["Señal"])
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()
    st.markdown("#### 🔍 Barrido de lag (opcional)")
    st.markdown(
        "Busca el desplazamiento temporal (*lag*) que maximiza la señal estadística. "
        "Útil cuando no sabes exactamente cuántas horas antes del evento ocurre el patrón."
    )

    with st.expander("Configurar y ejecutar barrido de lag"):
        col1, col2 = st.columns(2)
        with col1:
            max_lag = st.slider("Lag máximo (horas)", 12, 336, min(lookback_hours * 2, 144), step=12)
        with col2:
            lag_step = st.select_slider("Paso (horas)", options=[3, 6, 12, 24], value=6)

        if st.button("Ejecutar barrido de lag", type="secondary"):
            with st.spinner(f"Probando {len(range(0, max_lag + lag_step, lag_step))} lags..."):
                lag_results = run_lag_sweep(
                    events=events,
                    metric=metric,
                    max_lag=max_lag,
                    step=lag_step,
                    lookback_hours=lookback_hours,
                    baseline_days=baseline_days,
                    direction=direction,
                    baseline_strategy=baseline_strategy,
                    config=config,
                )
            st.session_state.lag_results = lag_results

    if st.session_state.get("lag_results"):
        lag_results = st.session_state.lag_results
        best = max(lag_results, key=lambda k: lag_results[k].association_strength)
        best_r = lag_results[best]
        st.success(
            f"Mejor lag encontrado: **{best}h** — "
            f"asociación={best_r.association_strength:.3f}, "
            f"p={best_r.p_value:.4f}, señal={best_r.signal_strength}"
        )
        st.plotly_chart(fig_lag_sweep(lag_results), use_container_width=True)


def render_tab_narrative(
    report: AlertReport,
    domain_cfg: dict,
    llm_provider: str,
    llm_api_key: str,
) -> None:
    tone = st.radio(
        "Tono de la narrativa",
        ["Divulgativo", "Técnico", "Profesional", "Neutral"],
        horizontal=True,
        help="Elige el estilo de redacción de la narrativa generada.",
    )

    st.markdown("### Narrativa automática (sin IA)")
    narrative_text = build_template_narrative(report, domain_cfg, tone)
    st.markdown(narrative_text)

    st.divider()
    st.markdown("### Narrativa con IA (opcional)")

    if report.level == "green" or not any(r.significant for r in report.results):
        st.info("No hay señales significativas para narrar con IA.")
        return

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"Usará **{llm_provider}** con la plantilla `{domain_cfg['prompt_template']}`. "
            "Requiere API key configurada en la barra lateral o en variable de entorno."
        )
    with col_btn:
        gen_btn = st.button("Generar con IA", type="primary")

    if gen_btn:
        with st.spinner("Generando narrativa con IA..."):
            narrated_report, err = try_llm_narration(
                report, llm_provider, llm_api_key, domain_cfg["prompt_template"]
            )

        if err:
            st.error(f"❌ {err}")
        else:
            has_narratives = any(r.narrative for r in narrated_report.results)
            if has_narratives:
                st.success("Narrativa generada correctamente.")
                for r in narrated_report.results:
                    if r.narrative:
                        st.markdown(f"**{r.metric_name}**: {r.narrative}")
                # Store for export
                st.session_state.report = narrated_report
            else:
                st.warning("El LLM no devolvió narrativa para ninguna señal significativa.")


def render_tab_export(
    report: AlertReport,
    metric: Metric,
    events: list[Event],
) -> None:
    st.markdown("### Exportar resultados")

    c1, c2, c3, c4 = st.columns(4)

    # JSON
    with c1:
        json_str = export_to_json(report, metric, events)
        st.download_button(
            label="⬇️ JSON",
            data=json_str,
            file_name="chrono_results.json",
            mime="application/json",
            use_container_width=True,
        )

    # HTML
    with c2:
        html_bytes = export_to_html_bytes(report)
        st.download_button(
            label="⬇️ HTML",
            data=html_bytes,
            file_name="chrono_report.html",
            mime="text/html",
            use_container_width=True,
        )

    # Markdown
    with c3:
        md_str = export_to_markdown_str(report)
        st.download_button(
            label="⬇️ Markdown",
            data=md_str,
            file_name="chrono_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # CSV de resultados
    with c4:
        csv_str = build_results_df(report).to_csv(index=False)
        st.download_button(
            label="⬇️ CSV",
            data=csv_str,
            file_name="chrono_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # CSV de datos de la métrica
    st.markdown("#### Datos de la serie temporal")
    df_metric = pd.DataFrame({"timestamp": metric.timestamps, "value": metric.values})
    st.download_button(
        label="⬇️ Descargar serie temporal (CSV)",
        data=df_metric.to_csv(index=False),
        file_name="chrono_metric.csv",
        mime="text/csv",
    )

    st.markdown("#### Eventos")
    df_events = pd.DataFrame([
        {"timestamp": e.timestamp.isoformat(), "label": e.label}
        for e in events
    ])
    st.download_button(
        label="⬇️ Descargar eventos (CSV)",
        data=df_events.to_csv(index=False),
        file_name="chrono_events.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("#### Vista previa del reporte HTML")
    with st.expander("Ver HTML"):
        st.code(html_bytes.decode("utf-8")[:4000] + "\n…", language="html")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render full sidebar; returns dict with all user choices."""
    st.sidebar.markdown("## ⏱️ Chrono Dashboard")
    st.sidebar.markdown("---")

    # Domain selector
    domain_labels = [f"{v['icon']} {k}" for k, v in DOMAINS.items()]
    sel_label = st.sidebar.selectbox("Dominio de análisis", domain_labels, index=0)
    domain_key = sel_label.split(" ", 1)[1]
    domain_cfg = DOMAINS[domain_key]

    st.sidebar.caption(domain_cfg["desc"])
    st.sidebar.markdown("---")

    # Data mode
    if domain_key == "Personalizado":
        mode = "upload"
    else:
        mode_opts = ["📊 Datos de ejemplo", "📂 Subir mis datos"]
        mode_sel  = st.sidebar.radio("Modo de datos", mode_opts)
        mode = "preset" if "ejemplo" in mode_sel else "upload"

    # File uploader (only in upload mode)
    uploaded_file = None
    if mode == "upload":
        st.sidebar.markdown("#### Subir datos")
        uploaded_file = st.sidebar.file_uploader(
            "CSV o Excel",
            type=["csv", "xlsx", "xls"],
            help="Columnas esperadas: fecha, valor numérico, etiqueta de evento (opcional).",
        )

    st.sidebar.markdown("---")

    # Analysis parameters
    st.sidebar.markdown("#### Parámetros de análisis")

    lookback_hours = st.sidebar.slider(
        "Ventana pre-evento (h)", 6, 720, domain_cfg["lookback_hours"], step=6,
        help="Horas antes de cada evento que se comparan contra el baseline.",
    )
    baseline_days = st.sidebar.slider(
        "Días de baseline", 7, 180, domain_cfg["baseline_days"],
        help="Días de historial que definen el comportamiento de referencia.",
    )

    direction = st.sidebar.selectbox(
        "Dirección de hipótesis",
        ["two-sided", "increase", "decrease"],
        index=["two-sided", "increase", "decrease"].index(domain_cfg["direction"]),
        help="¿Esperas que la métrica suba, baje, o cualquier dirección?",
    )
    baseline_strategy = st.sidebar.selectbox(
        "Estrategia de baseline",
        ["rolling", "same_weekday", "same_hour"],
        index=["rolling", "same_weekday", "same_hour"].index(domain_cfg["baseline_strategy"]),
        help=(
            "rolling: ventana deslizante. "
            "same_weekday: solo mismos días de semana. "
            "same_hour: solo misma hora del día."
        ),
    )

    # Advanced
    with st.sidebar.expander("⚙️ Opciones avanzadas"):
        correction_sel = st.selectbox("Corrección múltiple", ["fdr", "bonferroni", "ninguna"])
        correction = None if correction_sel == "ninguna" else correction_sel
        bootstrap_ci = st.checkbox(
            "Bootstrap CI (95%)", value=False,
            help="Calcula intervalos de confianza por bootstrap (~1s por métrica).",
        )
        alpha = st.slider("Significancia α", 0.01, 0.10, 0.05, step=0.01)
        strong_effect = st.slider("Umbral efecto fuerte", 0.10, 0.50, 0.25, step=0.05)

    # LLM
    with st.sidebar.expander("🤖 Narrativa con IA"):
        llm_provider = st.selectbox("Proveedor", ["anthropic", "groq", "ollama"])
        llm_api_key  = st.text_input(
            "API Key (opcional)", type="password",
            help="Si está vacío se lee la variable de entorno ANTHROPIC_API_KEY / GROQ_API_KEY.",
        )

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("▶ Ejecutar análisis", type="primary", use_container_width=True)

    return dict(
        domain_key=domain_key,
        domain_cfg=domain_cfg,
        mode=mode,
        uploaded_file=uploaded_file,
        lookback_hours=lookback_hours,
        baseline_days=baseline_days,
        direction=direction,
        baseline_strategy=baseline_strategy,
        correction=correction,
        bootstrap_ci=bootstrap_ci,
        alpha=alpha,
        strong_effect=strong_effect,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        run_btn=run_btn,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Header
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:14px;padding-bottom:6px">
          <span style="font-size:2.4rem;line-height:1">⏱️</span>
          <div>
            <h1 style="margin:0;font-size:1.9rem;line-height:1.1">Chrono Dashboard</h1>
            <p style="margin:0;color:#6b7280;font-size:.9rem">
              Correlación estadística entre series temporales y eventos discretos ·
              <a href="https://github.com/Raulcadiz/chrono-correlator" target="_blank">
              chrono-correlator 1.2.0</a>
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    opts = render_sidebar()
    domain_key   = opts["domain_key"]
    domain_cfg   = opts["domain_cfg"]
    mode         = opts["mode"]
    lookback_hours   = opts["lookback_hours"]
    baseline_days    = opts["baseline_days"]
    direction        = opts["direction"]
    baseline_strategy = opts["baseline_strategy"]
    correction       = opts["correction"]
    bootstrap_ci     = opts["bootstrap_ci"]
    alpha            = opts["alpha"]
    strong_effect    = opts["strong_effect"]
    llm_provider     = opts["llm_provider"]
    llm_api_key      = opts["llm_api_key"]
    run_btn          = opts["run_btn"]

    # Invalidate cached results when domain or mode changes
    cache_key = f"{domain_key}|{mode}"
    if st.session_state.get("_cache_key") != cache_key:
        st.session_state.pop("report", None)
        st.session_state.pop("lag_results", None)
        st.session_state["_cache_key"] = cache_key

    # ── Load / parse data ────────────────────────────────────────────────────
    metric: Optional[Metric] = None
    events: list[Event]      = []
    data_error: Optional[str] = None

    if mode == "preset":
        try:
            with st.spinner("Cargando datos de ejemplo..."):
                metric, events = load_preset_data(domain_key)
            st.success(
                f"Datos de ejemplo: **{len(metric.timestamps):,} puntos** · "
                f"**{len(events)} eventos** · "
                f"{metric.timestamps[0].strftime('%Y-%m-%d')} → "
                f"{metric.timestamps[-1].strftime('%Y-%m-%d')}"
            )
        except Exception as exc:
            data_error = str(exc)

    else:  # upload mode
        uploaded_file = opts["uploaded_file"]
        if uploaded_file is None:
            st.info("📂 Sube un archivo **CSV o Excel** en la barra lateral para comenzar.")
            st.markdown(
                """
**Formato esperado** (nombres de columna flexibles — se detectan automáticamente):

| fecha | valor | evento |
|-------|-------|--------|
| 2024-01-01 | 55.3 | |
| 2024-01-05 | 48.1 | Síntoma A |
| 2024-01-12 | 52.7 | |

La columna `evento` es **opcional**. Las filas con valor en esa columna se
convierten en eventos. Si no tienes columna de eventos, necesitas al menos
3 puntos con un patrón temporal para que el análisis sea significativo.
                """
            )
            return

        try:
            df = parse_uploaded_file(uploaded_file)
        except Exception as exc:
            st.error(f"❌ Error leyendo el archivo: {exc}")
            return

        st.success(f"Archivo cargado: **{uploaded_file.name}** — {len(df):,} filas · {len(df.columns)} columnas")

        with st.expander("Vista previa del archivo", expanded=True):
            st.dataframe(df.head(15), use_container_width=True)

        # Column mapping
        detected = detect_columns(df)
        st.markdown("#### Mapeo de columnas")
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            date_col = st.selectbox(
                "📅 Columna de fecha",
                all_cols,
                index=all_cols.index(detected["date"]) if detected["date"] in all_cols else 0,
            )
        with mc2:
            val_opts = num_cols if num_cols else all_cols
            val_idx  = val_opts.index(detected["value"]) if detected["value"] in val_opts else 0
            value_col = st.selectbox("📊 Columna de valor (numérica)", val_opts, index=val_idx)
        with mc3:
            evt_opts = ["(ninguna)"] + all_cols
            evt_idx  = evt_opts.index(detected["event"]) if detected["event"] in evt_opts else 0
            evt_sel  = st.selectbox("🏷️ Columna de eventos (opcional)", evt_opts, index=evt_idx)
            event_col = None if evt_sel == "(ninguna)" else evt_sel

        nc1, nc2 = st.columns(2)
        with nc1:
            metric_name = st.text_input("Nombre de la métrica", domain_cfg["metric_name"])
        with nc2:
            event_label_name = st.text_input("Nombre de los eventos", domain_cfg["event_label"])

        try:
            metric, events = build_metric_and_events(
                df, date_col, value_col, event_col, metric_name, event_label_name
            )
        except Exception as exc:
            data_error = f"Error procesando datos: {exc}"
        else:
            msg = (
                f"✅ Métrica: **{len(metric.timestamps):,} puntos** · "
                f"Eventos: **{len(events)}**"
            )
            if not events:
                msg += " — ⚠️ Sin eventos: el análisis requiere mínimo 3 eventos"
            st.info(msg)

    if data_error:
        st.error(f"❌ {data_error}")
        return

    if metric is None:
        return

    # ── Run analysis ─────────────────────────────────────────────────────────
    config = SignificanceConfig(alpha=alpha, strong_effect=strong_effect)

    should_run = run_btn or (mode == "preset" and "report" not in st.session_state)

    if should_run:
        if len(events) < 3:
            st.warning(
                f"El análisis requiere **al menos 3 eventos**. "
                f"Tienes {len(events)}. Ajusta la columna de eventos o usa datos de ejemplo."
            )
            return

        with st.spinner("Ejecutando análisis estadístico..."):
            try:
                report = run_analysis(
                    events=events,
                    metrics=[metric],
                    lookback_hours=lookback_hours,
                    baseline_days=baseline_days,
                    direction=direction,
                    baseline_strategy=baseline_strategy,
                    correction=correction,
                    bootstrap_ci=bootstrap_ci,
                    config=config,
                )
                st.session_state.report = report
                st.session_state.metric = metric
                st.session_state.events = events
                st.session_state.lag_results = None
            except Exception as exc:
                st.error(f"❌ Error en el análisis: {exc}")
                return

    report = st.session_state.get("report")
    if report is None:
        st.info("Haz clic en **▶ Ejecutar análisis** en la barra lateral para comenzar.")
        return

    s_metric = st.session_state.get("metric", metric)
    s_events = st.session_state.get("events", events)

    # ── Alert banner ──────────────────────────────────────────────────────────
    lc = LEVEL_COLOR[report.level]
    st.markdown(
        f"""
        <div style="background:{lc}18;border-left:4px solid {lc};
                    padding:10px 16px;border-radius:6px;margin:10px 0 18px">
          <b style="color:{lc};font-size:1rem">{LEVEL_LABEL[report.level]}</b>
          &nbsp;·&nbsp;
          <span style="color:#374151">
            {report.active_signals} señal(es) activa(s) de {report.total_signals} métrica(s) analizadas
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_d, tab_v, tab_r, tab_n, tab_e = st.tabs([
        "📋 Datos",
        "📊 Visualización",
        "📈 Resultados",
        "💬 Narrativa",
        "⬇️ Exportar",
    ])

    with tab_d:
        render_tab_datos(s_metric, s_events)

    with tab_v:
        render_tab_viz(s_metric, s_events, report, lookback_hours, baseline_days)

    with tab_r:
        render_tab_results(
            report, s_metric, s_events,
            lookback_hours, baseline_days, direction, baseline_strategy, config,
        )

    with tab_n:
        render_tab_narrative(report, domain_cfg, llm_provider, llm_api_key)

    with tab_e:
        render_tab_export(report, s_metric, s_events)


if __name__ == "__main__":
    main()
