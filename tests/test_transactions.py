"""Availability: the part of the data layer with real leakage risk.

An IL placement is routinely backdated. Filtering on the effective date would
give the model information that did not exist at prediction time -- genuine
look-ahead leakage, not merely optimism.
"""

import pandas as pd
import pytest

from mlb_pred.fetch_data.statsapi.transactions import (
    TRANSACTION_COLUMNS,
    _is_injury_related,
    transactions_known_as_of,
    transactions_strictly_before_date,
)


@pytest.fixture()
def transactions():
    return pd.DataFrame(
        [
            {
                "transaction_id": "1",
                "player_id": "100",
                "known_date": pd.Timestamp("2025-08-25").date(),
                "effective_date": pd.Timestamp("2025-08-22").date(),
                "is_backdated": True,
                "backdated_days": 3,
            },
            {
                "transaction_id": "2",
                "player_id": "200",
                "known_date": pd.Timestamp("2025-08-21").date(),
                "effective_date": pd.Timestamp("2025-08-21").date(),
                "is_backdated": False,
                "backdated_days": 0,
            },
        ]
    )


def test_backdated_transaction_is_invisible_before_it_was_announced(transactions):
    # The IL stint took effect on the 22nd but was announced on the 25th.
    # On the 23rd nobody knew, so the model must not either.
    on_23rd = transactions_known_as_of(transactions, "2025-08-23")
    assert set(on_23rd["transaction_id"]) == {"2"}


def test_backdated_transaction_becomes_visible_on_its_announcement_date(transactions):
    on_25th = transactions_known_as_of(transactions, "2025-08-25")
    assert set(on_25th["transaction_id"]) == {"1", "2"}


def test_historical_fallback_excludes_same_day_announcements(transactions):
    # The retrospective endpoint gives a date but no trustworthy announcement
    # time, so a game-day row is not safe for a pregame model.
    before_25th = transactions_strictly_before_date(transactions, "2025-08-25")
    assert set(before_25th["transaction_id"]) == {"2"}


def test_filtering_on_effective_date_would_have_leaked(transactions):
    # Documents the trap explicitly: the naive filter admits the row three
    # days early.
    naive = transactions[
        transactions["effective_date"] <= pd.Timestamp("2025-08-23").date()
    ]
    correct = transactions_known_as_of(transactions, "2025-08-23")
    assert len(naive) == 2
    assert len(correct) == 1


def test_empty_frame_passes_through(transactions):
    empty = transactions.iloc[0:0]
    assert transactions_known_as_of(empty, "2025-08-25").empty


def test_transaction_schema_keeps_both_dates():
    # If either date is dropped the leakage guard becomes unverifiable.
    assert "known_date" in TRANSACTION_COLUMNS
    assert "effective_date" in TRANSACTION_COLUMNS
    assert "is_backdated" in TRANSACTION_COLUMNS


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("placed RHP X on the 15-day injured list", True),
        ("placed C Y on the paternity list", True),
        ("placed Z on the bereavement list", True),
        ("optioned LHP W to Triple-A", False),
        ("signed free agent V", False),
    ],
)
def test_injury_related_detection(description, expected):
    assert _is_injury_related(description, "Status Change") is expected
