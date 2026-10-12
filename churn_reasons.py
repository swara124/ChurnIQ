"""
churn_reasons.py — per-customer churn reason generation using
feature importances + threshold-based rules on raw feature values.
"""
import numpy as np
import pandas as pd

# ── Reason rule definitions ──────────────────────────────────────────────────
# Each rule: (feature_keyword, condition_fn, reason_label, icon)
REASON_RULES = [
    # Balance
    ("balance",       lambda v, med: v < 0,                    "Negative Balance",      "💸"),
    ("balance",       lambda v, med: 0 <= v < med * 0.3,       "Very Low Balance",      "📉"),
    ("balance",       lambda v, med: med * 0.3 <= v < med * 0.6, "Below-Avg Balance",   "📊"),

    # Tenure
    ("tenure",        lambda v, med: v <= 1,                   "New Customer",          "🆕"),
    ("tenure",        lambda v, med: 2 <= v <= 3,              "Short Tenure",          "⏳"),

    # Age
    ("age",           lambda v, med: v > 58,                   "Senior Customer",       "👴"),
    ("age",           lambda v, med: v < 28,                   "Very Young Customer",   "🧑"),

    # Credit score
    ("creditscore",   lambda v, med: v < 500,                  "Poor Credit Score",     "⚠️"),
    ("creditscore",   lambda v, med: 500 <= v < 620,           "Low Credit Score",      "📋"),

    # Products
    ("numofproducts", lambda v, med: v == 1,                   "Single Product",        "📦"),
    ("numofproducts", lambda v, med: v >= 4,                   "Too Many Products",     "🔀"),

    # Activity
    ("isactivemember",lambda v, med: v == 0,                   "Inactive Member",       "😴"),

    # Campaign / contact pressure (bank dataset)
    ("campaign",      lambda v, med: v > med * 2,              "Over-Contacted",        "📞"),
    ("duration",      lambda v, med: v < med * 0.3,            "Low Call Engagement",   "📵"),
    ("pdays",         lambda v, med: v == -1,                  "Never Revisited",       "🔕"),
    ("previous",      lambda v, med: v == 0,                   "No Prior Contact",      "📭"),

    # Loan / housing
    ("loan",          lambda v, med: v == 1,                   "Has Personal Loan",     "💳"),
    ("housing",       lambda v, med: v == 1,                   "Has Housing Loan",      "🏠"),
    ("default",       lambda v, med: v == 1,                   "Credit Default",        "❗"),

    # Salary / estimated
    ("estimatedsalary", lambda v, med: v < med * 0.4,          "Low Income Estimate",   "💰"),
]


def get_reasons_for_customer(row: pd.Series, feature_importances: dict, top_n: int = 3) -> list:
    """
    Given a customer row and a dict {feature_name: importance_score},
    return up to top_n reasons as (icon, label) tuples.
    """
    triggered = []  # (importance, icon, label)

    for feat_key, condition, label, icon in REASON_RULES:
        # Find matching column in row
        matching = [c for c in row.index if feat_key in c.lower().replace("_", "")]
        for col in matching:
            val = row[col]
            if pd.isna(val):
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            # Median is encoded in condition via a placeholder; pass 0 if not available
            try:
                if condition(val, 0):
                    imp = feature_importances.get(col, 0.01)
                    triggered.append((imp, icon, label))
                    break  # only one trigger per rule group
            except Exception:
                pass

    # Sort by importance desc, deduplicate labels, take top_n
    triggered.sort(key=lambda x: -x[0])
    seen = set()
    result = []
    for imp, icon, label in triggered:
        if label not in seen:
            seen.add(label)
            result.append((icon, label))
        if len(result) >= top_n:
            break

    return result if result else [("🔍", "Behavioural Pattern")]


def build_reason_data(churn_df: pd.DataFrame, feature_importances: dict) -> list:
    """
    Returns list of dicts: {cid, risk, reasons: [(icon, label), ...]}
    Expects churn_df to have 'churn_risk_pct' and optionally id_column.
    """
    records = []
    for _, row in churn_df.iterrows():
        reasons = get_reasons_for_customer(row, feature_importances, top_n=3)
        records.append({
            "reasons": reasons,
        })
    return records


def get_reason_summary(all_reasons: list) -> list:
    """
    Aggregate reason frequency across all churn customers.
    Returns sorted list of (icon, label, count, pct).
    """
    from collections import Counter
    flat = []
    for r_list in all_reasons:
        for icon, label in r_list:
            flat.append(label)
    counts = Counter(flat)
    total = sum(counts.values()) or 1
    summary = [(label, count, round(count / total * 100, 1))
               for label, count in counts.most_common(8)]
    return summary


def get_medians(df: pd.DataFrame) -> dict:
    """Return median per numeric column."""
    return {c: df[c].median() for c in df.select_dtypes(include=[float, int]).columns}
