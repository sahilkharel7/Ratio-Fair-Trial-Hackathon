"""Wilson intervals for one rate and Newcombe's hybrid score interval for a difference of two rates,
checked against published values and the decision cases pinned in the build plan."""

import pytest

from ratio.modules.judge_stats import family_confidence, newcombe, wilson, z_value


def test_z_value_matches_the_normal_quantiles():
    assert z_value(0.95) == pytest.approx(1.959964, abs=1e-6)
    assert z_value(0.99) == pytest.approx(2.575829, abs=1e-6)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (5, 10, 0.2366, 0.7634),  # Brown, Cai and DasGupta (2001)
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (81, 263, 0.2553, 0.3662),  # Newcombe (1998), single proportion, Table I
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0, 0.1611),
        (1, 29, 0.0061, 0.1718),
    ],
)
def test_wilson_interval_matches_published_values(k, n, low, high):
    assert wilson(k, n, 0.95) == pytest.approx((low, high), abs=1e-4)


def test_wilson_interval_stays_within_zero_and_one():
    for n in range(1, 30):
        for k in range(n + 1):
            low, high = wilson(k, n, 0.999)
            assert 0.0 <= low <= k / n <= high <= 1.0


@pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n, 0.95)


@pytest.mark.parametrize(
    ("counts", "low", "high"),
    [  # Newcombe (1998), Statistics in Medicine 17:873-890, Table II, method 10
        ((56, 70, 48, 80), 0.0524, 0.3339),
        ((9, 10, 3, 10), 0.1705, 0.8090),
        ((6, 7, 2, 7), 0.0582, 0.8062),
        ((5, 56, 0, 29), -0.0381, 0.1926),
    ],
)
def test_newcombe_interval_matches_the_published_examples(counts, low, high):
    assert newcombe(*counts, 0.95) == pytest.approx((low, high), abs=1e-4)


def test_newcombe_interval_is_antisymmetric():
    low, high = newcombe(9, 10, 3, 10, 0.95)
    assert newcombe(3, 10, 9, 10, 0.95) == pytest.approx((-high, -low))


def test_family_confidence_splits_alpha_across_the_indicators_compared():
    assert family_confidence(0.05, 1) == pytest.approx(0.95)
    assert family_confidence(0.05, 5) == pytest.approx(0.99)
    with pytest.raises(ValueError):
        family_confidence(0.05, 0)


def test_plan_decision_cases_seven_of_seven_fires_six_of_seven_does_not():
    confidence = family_confidence(0.05, 6)
    low, _ = newcombe(7, 7, 1, 6, confidence)
    assert low == pytest.approx(0.12, abs=0.01) and low > 0
    low, _ = newcombe(6, 7, 1, 6, confidence)
    assert low < 0


def test_demo_design_detention_fires_and_the_other_two_do_not():
    confidence = family_confidence(0.05, 3)  # three indicators are shown on the demo judge's page
    assert newcombe(9, 9, 4, 10, confidence)[0] > 0
    assert newcombe(8, 9, 7, 10, confidence)[0] < 0
    assert newcombe(5, 9, 2, 10, confidence)[0] < 0
