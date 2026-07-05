# 📊 Customer Support Analytics & Escalation Risk Dashboard

An end-to-end data analyst workflow — from raw data understanding to dashboarding
and ML-driven insight — on the public Kaggle **Customer Support Ticket Dataset**.

## Project structure

```
customer_support_tickets.csv   raw Kaggle export (input, not modified)
data_prep.py                   cleaning + feature engineering -> data/processed_tickets.csv
model.py                       leakage-free logistic regression -> data/tickets_scored.csv
app.py                         Streamlit dashboard (KPIs, bottlenecks, ML risk view)
requirements.txt
data/
  processed_tickets.csv        cleaned + engineered ticket-level table
  tickets_scored.csv           processed table + escalation_risk_score + risk_category
                                (also the file to import into Tableau Public)
```

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python data_prep.py     # -> data/processed_tickets.csv
python model.py         # -> data/tickets_scored.csv, prints model evaluation
streamlit run app.py    # opens the dashboard in your browser
```

For Tableau Public: open `data/tickets_scored.csv` (or use the dashboard's
sidebar download button) as a data source and build visuals against
`priority`, `ticket_type`, `ticket_channel`, `resolution_hours`, `sla_breach`,
`escalation_risk_score`, and `risk_category`.

## ⚠️ Data quality finding — read this first

The brief assumed `time_to_resolution` is a clean hours-to-resolve field.
**It isn't, in this dataset.** `First Response Time` and `Time to Resolution`
are Faker-generated timestamps that both fall on 2023-06-01, with a gap that
is sometimes *negative* (resolution recorded before first response) and shows
no relationship to priority, ticket type, or the satisfaction rating. Only
the categorical fields — priority, ticket type, channel, product, customer
age — carry real structure; the timing and satisfaction fields are
randomized filler. This is a known characteristic of this particular public
dataset, not a bug in the pipeline here.

Consequence: the escalation model, trained honestly on this data, scores
**0.51 ROC AUC** (0.50 = chance) — it correctly finds no usable signal.
Rather than manufacture a cleaner story, the pipeline is built exactly as it
would be for real data (see below), and the dashboard states this finding
up front. The result itself is the deliverable: a real analyst should catch
and report this rather than silently present numbers computed on noise as
operational fact.

## Data understanding & assumptions

- SLA threshold: 48 hours.
- No true ticket-creation timestamp exists in the raw data; `First Response
  Time` is used as the `created_time` proxy (the earliest real event captured
  per ticket).
- `resolution_hours` = `Time to Resolution` − `First Response Time`, only
  defined for `Closed` tickets (the other two statuses have no resolution
  timestamp). The dashboard reports its absolute value to avoid a
  meaningless negative average — see the data quality note above.
- There is no ground-truth "escalated" column. It's derived, for `Closed`
  tickets only, as: **SLA breach OR a resolved ticket the customer still
  rated ≤2/5**.

## KPI design

| KPI | Why it matters |
|---|---|
| Total Tickets | Workload volume |
| Avg Resolution Time | Efficiency (see data quality caveat) |
| SLA Breach Rate | Service reliability |
| Escalation Rate | Operational risk |

## Machine learning layer

- **Problem**: binary classification — predict probability of escalation.
- **Features**: priority, ticket type, channel, customer age — deliberately
  limited to information known **at ticket creation**, not at resolution.
  `resolution_hours` and `customer_satisfaction_score` are excluded from the
  features because they are the ingredients of the label itself; including
  them would leak the outcome into the input.
- **Why this matters**: it lets the same model score currently `Open` /
  `Pending Customer Response` tickets, not just already-`Closed` ones — the
  actual business use case (flag risk *before* resolution, not after).
- **Model**: Logistic Regression (`class_weight="balanced"`), chosen for
  interpretability over accuracy.
- **Output**: `escalation_risk_score` (0–1). Because the model's predicted
  probabilities cluster tightly around the base rate (no real signal in this
  data), `risk_category` is assigned by **tercile of predicted score**
  (bottom/middle/top third = Low/Medium/High) rather than fixed probability
  cutoffs — a relative ranking a support lead can still triage against, not
  a calibrated risk probability.

## Dashboard structure (`app.py`)

1. **Service Health Overview** — Total Tickets, Avg Resolution Time, SLA
   Breach Rate, Escalation Rate.
2. **Bottleneck Analysis** — avg resolution time by priority and by ticket
   type.
3. **ML-Based Risk View** — risk distribution + a sortable table of
   currently High-risk tickets.
4. **Key Insights** — computed live from the filtered data (not hardcoded),
   including the model's honest performance finding.

Sidebar filters: priority, ticket type, channel, status. A download button
exports the fully enriched CSV for Tableau Public.

## Tech stack

- Python — data preprocessing & ML (pandas, scikit-learn)
- Streamlit + Plotly — interactive dashboard
- Tableau Public — Alternative dashboard from the exported CSV
