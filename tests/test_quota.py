import pytest

from bench_core.quota import QuotaConfig, QuotaState


def test_quota_debit_refund_and_snapshot() -> None:
    config = QuotaConfig.from_dict(
        {"train_samples": {"limit": 100, "terminal": True}}
    )
    state = QuotaState(config)

    state.debit("train_samples", 80)
    state.debit("train_samples", 50)
    assert state.snapshot()["train_samples"] == {
        "limit": 100.0,
        "spent": 100.0,
        "remaining": 0.0,
        "exhausted": True,
        "terminal": True,
    }

    state.refund("train_samples", 25)
    assert state.remaining("train_samples") == 25


def test_quota_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        QuotaConfig.from_dict({"calls": {"limit": -1}})

    state = QuotaState(QuotaConfig.from_dict({"calls": {"limit": 1}}))
    with pytest.raises(ValueError):
        state.debit("calls", -1)

