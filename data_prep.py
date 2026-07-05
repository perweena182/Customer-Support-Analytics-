"""
Data preparation for the Customer Support Analytics & Escalation Risk project.

Assumptions (see README for full discussion):
- SLA threshold: 48 hours.
- The raw Kaggle dataset has no true ticket-creation timestamp. "First Response
  Time" is the earliest real event captured per ticket, so it is used as the
  created_time proxy.
- "Time to Resolution" and "First Response Time" in this dataset are randomized
  Faker timestamps (both fall on 2023-06-01, and the delta between them is
  sometimes negative). They do NOT represent a real elapsed handling duration.
  resolution_hours / sla_breach are still computed exactly as specified, but
  are documented as illustrative of methodology rather than real SLA fact --
  see the "Data Quality Notice" in the dashboard and README.
- Escalation has no ground-truth label in the raw data. It is derived, for
  tickets that are already Closed, as: SLA breach OR a resolved ticket the
  customer still rated <=2/5.
"""
import os

import numpy as np
import pandas as pd

RAW_PATH = "customer_support_tickets.csv"
OUT_PATH = "data/processed_tickets.csv"
SLA_HOURS = 48


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    first_response = pd.to_datetime(df["First Response Time"], errors="coerce")
    resolved_time = pd.to_datetime(df["Time to Resolution"], errors="coerce")

    resolution_hours = (resolved_time - first_response).dt.total_seconds() / 3600
    sla_breach = np.where(resolution_hours.notna(), resolution_hours > SLA_HOURS, np.nan)
    satisfaction = df["Customer Satisfaction Rating"]

    escalated = np.where(
        resolution_hours.notna(),
        ((sla_breach == True) | (satisfaction <= 2)).astype(float),  # noqa: E712
        np.nan,
    )

    out = pd.DataFrame(
        {
            "ticket_id": df["Ticket ID"],
            "priority": df["Ticket Priority"],
            "ticket_type": df["Ticket Type"],
            "ticket_channel": df["Ticket Channel"],
            "ticket_status": df["Ticket Status"],
            "customer_age": df["Customer Age"],
            "customer_gender": df["Customer Gender"],
            "product_purchased": df["Product Purchased"],
            "created_time": first_response,
            "resolved_time": resolved_time,
            "resolution_hours": resolution_hours,
            "sla_breach": sla_breach,
            "customer_satisfaction_score": satisfaction,
            "escalated": escalated,
        }
    )
    return out


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    processed = prepare(load_raw())
    processed.to_csv(OUT_PATH, index=False)

    n_labeled = processed["escalated"].notna().sum()
    print(f"Wrote {len(processed)} rows to {OUT_PATH}")
    print(f"{n_labeled} tickets are Closed and have a usable escalation label.")
    print(f"Escalation rate among labeled tickets: {processed['escalated'].mean():.1%}")
