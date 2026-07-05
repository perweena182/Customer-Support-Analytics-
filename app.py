"""
Customer Support Analytics & Escalation Risk Dashboard.

Run with:  streamlit run app.py
Requires data/tickets_scored.csv, produced by:  python data_prep.py && python model.py
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "data/tickets_scored.csv"
SLA_HOURS = 48

# --- Validated palette (dataviz skill reference instance) -----------------
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

CAT_BLUE = "#2a78d6"
PRIORITY_SEQ = {"Low": "#9ec5f4", "Medium": "#5598e7", "High": "#2a78d6", "Critical": "#184f95"}
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_ORDER = ["Low", "Medium", "High"]

CHART_LAYOUT = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font_color=INK_PRIMARY,
    margin=dict(t=50, b=20, l=10, r=10),
    yaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE),
    xaxis=dict(gridcolor=GRIDLINE),
)

st.set_page_config(
    page_title="Customer Support Analytics & Escalation Risk",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["created_time", "resolved_time"])
    return df


if not os.path.exists(DATA_PATH):
    st.error(
        "No scored data found. Run these first from the project folder:\n\n"
        "```\npython data_prep.py\npython model.py\n```"
    )
    st.stop()

df = load_data()

st.title("📊 Customer Support Analytics & Escalation Risk Dashboard")
st.caption(
    "Raw tickets → KPIs → bottleneck analysis → ML escalation risk scoring, "
    "on the public Kaggle Customer Support Ticket Dataset."
)

with st.expander("⚠️ Data quality notice — read before trusting the numbers below"):
    st.markdown(
        """
This dataset's `First Response Time` and `Time to Resolution` fields are
**synthetically generated and not real elapsed durations** — both timestamps
fall on the same day, their gap is sometimes negative (resolution recorded
*before* first response), and it shows no relationship to priority, ticket
type, or satisfaction rating. The **escalation model reflects this**: it
scores **0.51 ROC AUC** (0.50 = pure chance) on held-out data — it found no
usable signal in the categorical features (priority, type, channel, age)
either.

The pipeline below is built exactly as it would be for a real ticketing
dataset — KPIs, bottleneck breakdowns, leakage-free feature/label separation,
train/test evaluation — and would produce genuine insight on data with real
signal. On *this* dataset, treat the numbers as a methodology demonstration,
not an operational finding. Risk categories are **relative rank** (top/bottom
third of predicted scores), not a calibrated probability, since the model's
predicted probabilities cluster tightly around the base rate.
"""
    )

st.sidebar.header("Filters")
priorities = st.sidebar.multiselect(
    "Priority", sorted(df["priority"].unique()), default=sorted(df["priority"].unique())
)
types = st.sidebar.multiselect(
    "Ticket type", sorted(df["ticket_type"].unique()), default=sorted(df["ticket_type"].unique())
)
channels = st.sidebar.multiselect(
    "Channel", sorted(df["ticket_channel"].unique()), default=sorted(df["ticket_channel"].unique())
)
statuses = st.sidebar.multiselect(
    "Status", sorted(df["ticket_status"].unique()), default=sorted(df["ticket_status"].unique())
)

fdf = df[
    df["priority"].isin(priorities)
    & df["ticket_type"].isin(types)
    & df["ticket_channel"].isin(channels)
    & df["ticket_status"].isin(statuses)
]

st.sidebar.divider()
st.sidebar.download_button(
    "⬇️ Download enriched CSV (for Tableau)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="tickets_enriched_for_tableau.csv",
    mime="text/csv",
    help="All tickets with engineered features + escalation_risk_score + risk_category.",
)

st.divider()

# --- 1. Service Health Overview --------------------------------------------
st.subheader("1️⃣ Service Health Overview")
st.caption("How healthy is the support operation?")

closed = fdf.dropna(subset=["resolution_hours"])
total_tickets = len(fdf)
avg_resolution = closed["resolution_hours"].abs().mean() if len(closed) else float("nan")
sla_breach_rate = closed["sla_breach"].astype(bool).mean() if len(closed) else float("nan")
escalation_rate = closed["escalated"].astype(bool).mean() if len(closed) else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total tickets", f"{total_tickets:,}")
c2.metric(
    "Avg resolution time",
    f"{avg_resolution:.1f} hrs" if pd.notna(avg_resolution) else "—",
    help="Mean |Time to Resolution − First Response Time| among Closed tickets.",
)
c3.metric(
    "SLA breach rate",
    f"{sla_breach_rate:.0%}" if pd.notna(sla_breach_rate) else "—",
    help=f"Share of Closed tickets whose resolution time exceeded the {SLA_HOURS}h SLA.",
)
c4.metric(
    "Escalation rate",
    f"{escalation_rate:.0%}" if pd.notna(escalation_rate) else "—",
    help="Share of Closed tickets that breached SLA or were rated ≤2/5.",
)

st.divider()

# --- 2. Bottleneck Analysis --------------------------------------------------
st.subheader("2️⃣ Bottleneck Analysis")
st.caption("Where are delays coming from?")

col1, col2 = st.columns(2)

by_priority = (
    closed.assign(abs_hours=closed["resolution_hours"].abs())
    .groupby("priority")["abs_hours"]
    .mean()
    .reindex([p for p in PRIORITY_ORDER if p in priorities])
    .dropna()
)
fig_priority = go.Figure(
    go.Bar(
        x=by_priority.index,
        y=by_priority.values,
        marker_color=[PRIORITY_SEQ.get(p, CAT_BLUE) for p in by_priority.index],
        text=[f"{v:.1f}h" for v in by_priority.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}h<extra></extra>",
    )
)
fig_priority.update_layout(title="Avg resolution time by priority", yaxis_title="Hours", **CHART_LAYOUT)
col1.plotly_chart(fig_priority, use_container_width=True)

by_type = (
    closed.assign(abs_hours=closed["resolution_hours"].abs())
    .groupby("ticket_type")["abs_hours"]
    .mean()
    .sort_values(ascending=False)
)
fig_type = go.Figure(
    go.Bar(
        x=by_type.index,
        y=by_type.values,
        marker_color=CAT_BLUE,
        text=[f"{v:.1f}h" for v in by_type.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}h<extra></extra>",
    )
)
fig_type.update_layout(title="Avg resolution time by ticket type", yaxis_title="Hours", **CHART_LAYOUT)
col2.plotly_chart(fig_type, use_container_width=True)

st.divider()

# --- 3. ML-Based Risk View ---------------------------------------------------
st.subheader("3️⃣ ML-Based Risk View")
st.caption("Which tickets need a first look? Scored on ALL tickets, including still-open ones.")

col3, col4 = st.columns([1, 2])

risk_counts = fdf["risk_category"].value_counts().reindex(RISK_ORDER).fillna(0)
fig_risk = go.Figure(
    go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker_color=[STATUS[r] for r in risk_counts.index],
        text=[f"{int(v):,}" for v in risk_counts.values],
        textposition="outside",
        hovertemplate="%{x} risk: %{y:,} tickets<extra></extra>",
    )
)
fig_risk.update_layout(title="Risk distribution", yaxis_title="Tickets", **CHART_LAYOUT)
col3.plotly_chart(fig_risk, use_container_width=True)

with col4:
    st.markdown("**High-risk tickets — focus list**")
    high_risk = fdf[fdf["risk_category"] == "High"].sort_values(
        "escalation_risk_score", ascending=False
    )
    display = high_risk[
        ["ticket_id", "priority", "ticket_type", "ticket_channel", "ticket_status", "escalation_risk_score"]
    ].copy()
    display["escalation_risk_score"] = (display["escalation_risk_score"] * 100).round(1)
    display = display.rename(columns={"escalation_risk_score": "risk_score_%"})
    st.dataframe(display, use_container_width=True, height=320, hide_index=True)

st.divider()

# --- 4. Key Insights ----------------------------------------------------------
st.subheader("🧪 Key Insights")

still_open_high_risk = int(
    ((fdf["risk_category"] == "High") & fdf["resolution_hours"].isna()).sum()
)
worst_type = by_type.idxmax() if len(by_type) else "—"
worst_type_hours = by_type.max() if len(by_type) else float("nan")
spread = (by_priority.max() - by_priority.min()) if len(by_priority) > 1 else float("nan")

st.markdown(
    f"""
- **{worst_type}** has the longest average resolution time among ticket types
  in the current filter (**{worst_type_hours:.1f}h**), {"a meaningful gap" if pd.notna(spread) and spread > 1 else "though the spread across categories is small"} —
  consistent with the finding above that this dataset's timing fields carry
  little real signal.
- **{still_open_high_risk:,}** currently Open/Pending tickets fall in the
  **High** relative-risk third — these are the ones a support lead would
  triage first under this methodology, ahead of any SLA breach occurring.
- The model's near-chance AUC (0.51) is itself an insight: on this dataset,
  **priority, ticket type, and channel alone don't predict which resolved
  tickets breached SLA or scored low satisfaction** — a real deployment would
  need better signal (agent workload, ticket text/sentiment, queue depth) to
  do meaningfully better than random.
"""
)
