"""
Credit card default: a bank-style credit scorecard and expected credit loss.

Data: 30,000 credit card clients in Taiwan, April-September 2005
(Yeh, I. (2009), UCI Machine Learning Repository, CC BY 4.0).

The pipeline:
  1. Build behavioural characteristics (repayment status, utilisation, ...)
  2. Group each characteristic into bins and measure its power (WoE, IV)
  3. Fit a logistic regression on the WoE values
  4. Turn the model into a points-based scorecard
  5. Validate it (AUC, Gini, KS, calibration)
  6. Use it: probability of default -> expected credit loss, approval cut-offs
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

TARGET = "default"

# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Read the UCI file (the .xls, or the .zip it comes in)."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".xls"))
            with z.open(name) as f:
                df = pd.read_excel(f, header=1)
    elif path.suffix == ".xls":
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"}).copy()
    df.columns = [c.upper() if c != TARGET else c for c in df.columns]
    # Codes not in the data dictionary are grouped into "other"
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 4), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].between(1, 3), 3)
    return df.set_index("ID")


PAY = [f"PAY_{i}" for i in range(1, 7)]           # repayment status, Sep (1) back to Apr (6)
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]       # bill statement amount
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]        # amount actually paid

DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


# ---------------------------------------------------------------------------
# 2. Behavioural characteristics (no demographics: see the notebook for why)
# ---------------------------------------------------------------------------

def build_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """The five characteristics the scorecard is built on."""
    out = pd.DataFrame(index=df.index)

    latest = df["PAY_1"]
    out["latest_status"] = np.select(
        [latest <= -2, latest == -1, latest == 0, latest == 1],
        ["a. No balance", "b. Paid in full", "c. Paid minimum", "d. 1 month late"],
        default="e. 2+ months late")

    late = (df[PAY] > 0).sum(axis=1)
    out["months_late_6m"] = pd.cut(late, [-1, 0, 1, 2, 6], labels=["0", "1", "2", "3-6"]).astype(str)

    out["credit_limit"] = df["LIMIT_BAL"]

    ratio = (df["PAY_AMT2"] / df["BILL_AMT2"]).where(df["BILL_AMT2"] > 0)   # share of last month's bill repaid
    out["repayment_ratio"] = ratio.clip(0, 1)

    out["avg_util_6m"] = (df[BILL].clip(lower=0).mean(axis=1) / df["LIMIT_BAL"]).clip(upper=1.5)
    return out


CATEGORICAL = ["latest_status", "months_late_6m"]
NUMERIC = ["credit_limit", "repayment_ratio", "avg_util_6m"]


# ---------------------------------------------------------------------------
# 3. Binning, Weight of Evidence and Information Value
# ---------------------------------------------------------------------------

def fit_bins(x: pd.Series, n_bins: int = 5) -> list[float]:
    """Quantile cut points for a numeric characteristic (fitted on training data)."""
    qs = np.unique(np.nanquantile(x.dropna(), np.linspace(0, 1, n_bins + 1)))
    edges = [-np.inf, *qs[1:-1], np.inf]
    return edges


def apply_bins(x: pd.Series, edges: list[float]) -> pd.Series:
    labels = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        fmt = (lambda v: f"{v:,.0f}") if max(abs(e) for e in edges if np.isfinite(e)) > 100 else (lambda v: f"{v:.2f}")
        lo_s = "min" if np.isinf(lo) else fmt(lo)
        hi_s = "max" if np.isinf(hi) else fmt(hi)
        labels.append(f"{i}. {lo_s} to {hi_s}")
    binned = pd.cut(x, edges, labels=labels, include_lowest=True).astype(str)
    return binned.where(x.notna(), "Missing")


def woe_table(binned: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Weight of Evidence per bin, and the characteristic's Information Value.

    WoE = ln(share of all goods in the bin / share of all bads in the bin)
    Positive WoE = safer than average; negative = riskier.
    """
    t = pd.crosstab(binned, y)
    t.columns = ["good", "bad"]
    t = t + 0.5                                  # avoids log(0) in tiny bins
    t["count"] = t["good"] + t["bad"] - 1
    t["default_rate"] = (t["bad"] - 0.5) / t["count"]
    t["dist_good"] = t["good"] / t["good"].sum()
    t["dist_bad"] = t["bad"] / t["bad"].sum()
    t["woe"] = np.log(t["dist_good"] / t["dist_bad"])
    t["iv"] = (t["dist_good"] - t["dist_bad"]) * t["woe"]
    return t[["count", "default_rate", "woe", "iv"]]


def iv_strength(iv: float) -> str:
    """The usual rule of thumb banks use for Information Value."""
    if iv < 0.02:
        return "Useless"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Very strong (check for leakage)"


class WoEEncoder:
    """Learns bins + WoE on training data, then applies them to any data."""

    def fit(self, chars: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        self.edges_ = {c: fit_bins(chars[c]) for c in NUMERIC}
        self.tables_ = {c: woe_table(self._bin(chars, c), y) for c in chars.columns}
        self.iv_ = pd.Series({c: t["iv"].sum() for c, t in self.tables_.items()}).sort_values(ascending=False)
        return self

    def _bin(self, chars: pd.DataFrame, c: str) -> pd.Series:
        return apply_bins(chars[c], self.edges_[c]) if c in NUMERIC else chars[c].astype(str)

    def bins(self, chars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self._bin(chars, c) for c in self.tables_})

    def transform(self, chars: pd.DataFrame) -> pd.DataFrame:
        b = self.bins(chars)
        return pd.DataFrame({c: b[c].map(self.tables_[c]["woe"]).fillna(0.0) for c in b})


# ---------------------------------------------------------------------------
# 4. Model and scorecard
# ---------------------------------------------------------------------------

def fit_logistic(woe: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=1000).fit(woe, y)   # (almost) unpenalised


# Scaling convention: 600 points = odds of 50 good : 1 bad; every 20 points doubles the odds
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def scorecard(enc: WoEEncoder, model: LogisticRegression) -> pd.DataFrame:
    """Points for every bin of every characteristic. A client's score is the sum."""
    n = len(enc.tables_)
    rows = []
    for coef, c in zip(model.coef_[0], enc.tables_):
        for b, r in enc.tables_[c].iterrows():
            # ln(odds good) = -(intercept + sum coef*woe), so points rise as risk falls
            pts = -(coef * r["woe"] + model.intercept_[0] / n) * FACTOR + OFFSET / n
            rows.append({"characteristic": c, "bin": b, "clients": int(r["count"]),
                         "default_rate": r["default_rate"], "woe": r["woe"], "points": round(pts)})
    return pd.DataFrame(rows)


def score(enc: WoEEncoder, card: pd.DataFrame, chars: pd.DataFrame) -> pd.Series:
    b = enc.bins(chars)
    total = pd.Series(0.0, index=chars.index)
    for c in b:
        pts = card[card["characteristic"] == c].set_index("bin")["points"]
        total += b[c].map(pts).fillna(0)
    return total.rename("score")


def score_to_pd(s: pd.Series | float) -> pd.Series | float:
    """Invert the scaling: score -> probability of default."""
    odds_good = np.exp((s - OFFSET) / FACTOR)
    return 1 / (1 + odds_good)


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

def gini(y, p) -> float:
    """Gini = 2 x AUC - 1; the standard ranking measure in credit risk."""
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    """Kolmogorov-Smirnov: biggest gap between cumulative bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def calibration_table(y: pd.Series, p: pd.Series, groups: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["group"] = pd.qcut(d["p"].rank(method="first"), groups, labels=range(1, groups + 1))
    return d.groupby("group", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), clients=("y", "size"))


# ---------------------------------------------------------------------------
# 6. Expected credit loss
# ---------------------------------------------------------------------------

LGD = 0.75   # loss given default for unsecured card debt (assumption)
CCF = 0.50   # share of the unused limit drawn before default (assumption)


def exposure_at_default(df: pd.DataFrame, ccf: float = CCF) -> pd.Series:
    """EAD = current balance + CCF x unused limit."""
    balance = df["BILL_AMT1"].clip(lower=0)
    unused = (df["LIMIT_BAL"] - balance).clip(lower=0)
    return balance + ccf * unused


def expected_loss(pd_: pd.Series, ead: pd.Series, lgd: float = LGD) -> pd.Series:
    """EL = PD x LGD x EAD."""
    return pd_ * lgd * ead
