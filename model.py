"""
Escalation risk model.

Design note: the escalated label (see data_prep.py) is derived from resolution
outcomes (SLA breach / low satisfaction), which are only known AFTER a ticket
closes. Training on resolution_hours or satisfaction as features would leak
the label into the inputs. Instead, the model uses only information available
the moment a ticket is opened -- priority, ticket type, channel, customer age
-- so it can score currently Open/Pending tickets too. That mirrors the real
business use case: flagging risk before a ticket resolves, not after.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

IN_PATH = "data/processed_tickets.csv"
OUT_PATH = "data/tickets_scored.csv"

CATEGORICAL_FEATURES = ["priority", "ticket_type", "ticket_channel"]
NUMERIC_FEATURES = ["customer_age"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def risk_bucket(score: float, low_cut: float, high_cut: float) -> str:
    if score >= high_cut:
        return "High"
    if score >= low_cut:
        return "Medium"
    return "Low"


def main() -> None:
    df = pd.read_csv(IN_PATH)

    labeled = df.dropna(subset=["escalated"]).copy()
    X, y = labeled[FEATURES], labeled["escalated"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print("=== Model evaluation (held-out closed tickets) ===")
    print(f"ROC AUC: {roc_auc_score(y_test, test_probs):.3f}  (0.5 = no better than chance)")
    print(classification_report(y_test, test_preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_preds))

    df["escalation_risk_score"] = pipeline.predict_proba(df[FEATURES])[:, 1]

    # With AUC ~0.51 the model has effectively no signal in this dataset (see
    # README "Data Quality Notice"), so predicted probabilities cluster tightly
    # around the base rate and fixed thresholds (e.g. >=0.66) would put every
    # ticket in one bucket. Tercile-based buckets instead give a relative rank
    # -- "look at these first" -- which is the one thing a weak-but-nonzero-
    # signal model can still offer, rather than a calibrated probability.
    low_cut, high_cut = df["escalation_risk_score"].quantile([1 / 3, 2 / 3])
    df["risk_category"] = df["escalation_risk_score"].apply(
        lambda s: risk_bucket(s, low_cut, high_cut)
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} scored rows to {OUT_PATH}")
    print(f"Risk cutoffs -- low/high: {low_cut:.4f} / {high_cut:.4f}")
    print(df["risk_category"].value_counts())


if __name__ == "__main__":
    main()
