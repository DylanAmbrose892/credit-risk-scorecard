import numpy as np
import pandas as pd
import pytest

from creditrisk import core as cr


def test_scaling_anchor_points():
    # 600 points means odds of 50 good : 1 bad, so PD = 1/51
    assert cr.score_to_pd(600) == pytest.approx(1 / 51)
    # every 20 points doubles the odds of being good
    odds = lambda s: (1 - cr.score_to_pd(s)) / cr.score_to_pd(s)
    assert odds(620) / odds(600) == pytest.approx(2)


def test_woe_signs_and_iv():
    binned = pd.Series(["safe"] * 100 + ["risky"] * 100)
    y = pd.Series([0] * 90 + [1] * 10 + [0] * 40 + [1] * 60)
    t = cr.woe_table(binned, y)
    assert t.loc["safe", "woe"] > 0 > t.loc["risky", "woe"]
    assert t.loc["safe", "default_rate"] == pytest.approx(0.10)
    assert t["iv"].sum() > 0.5


def test_iv_rule_of_thumb():
    assert cr.iv_strength(0.01) == "Useless"
    assert cr.iv_strength(0.2) == "Medium"


def test_clean_groups_unknown_codes():
    raw = pd.DataFrame({"ID": [1, 2], "EDUCATION": [0, 2], "MARRIAGE": [0, 1],
                        "PAY_0": [0, 1], "default payment next month": [0, 1]})
    df = cr.clean(raw)
    assert df["EDUCATION"].tolist() == [4, 2]
    assert df["MARRIAGE"].tolist() == [3, 1]
    assert "PAY_1" in df and "default" in df


def test_exposure_and_expected_loss():
    df = pd.DataFrame({"BILL_AMT1": [40_000, -500], "LIMIT_BAL": [100_000, 50_000]})
    ead = cr.exposure_at_default(df, ccf=0.5)
    assert ead.tolist() == [70_000, 25_000]      # 40k + 0.5 x 60k ; credit balance counts as 0
    el = cr.expected_loss(pd.Series([0.1, 0.2]), ead, lgd=0.75)
    assert el.tolist() == pytest.approx([5_250, 3_750])


def test_scorecard_reproduces_model():
    rng = np.random.default_rng(0)
    n = 4000
    chars = pd.DataFrame({
        "latest_status": rng.choice(["a. No balance", "e. 2+ months late"], n),
        "months_late_6m": rng.choice(["0", "3-6"], n),
        "credit_limit": rng.uniform(10_000, 500_000, n),
        "repayment_ratio": rng.uniform(0, 1, n),
        "avg_util_6m": rng.uniform(0, 1, n),
    })
    logit = -1.5 + 2.0 * (chars["latest_status"] == "e. 2+ months late") - chars["credit_limit"] / 300_000
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    enc = cr.WoEEncoder().fit(chars, y)
    model = cr.fit_logistic(enc.transform(chars), y)
    card = cr.scorecard(enc, model)
    pd_from_points = cr.score_to_pd(cr.score(enc, card, chars))
    pd_from_model = model.predict_proba(enc.transform(chars))[:, 1]
    # points are rounded to whole numbers, so allow a small difference
    assert np.abs(pd_from_points - pd_from_model).max() < 0.03
