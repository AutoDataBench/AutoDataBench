from bench_core.snapshot import BestSoFarSnapshot


def test_snapshot_keeps_best_maximum() -> None:
    snapshot = BestSoFarSnapshot()

    assert snapshot.consider(dataset_path="first.jsonl", score=0.4, dataset_hash="a")
    assert not snapshot.consider(
        dataset_path="worse.jsonl", score=0.3, dataset_hash="b"
    )
    assert snapshot.consider(dataset_path="best.jsonl", score=0.6, dataset_hash="c")
    assert snapshot.dataset_path == "best.jsonl"
    assert snapshot.updates == 2


def test_snapshot_can_minimize() -> None:
    snapshot = BestSoFarSnapshot(minimize=True)

    assert snapshot.consider(dataset_path="first.jsonl", score=2.0, dataset_hash="a")
    assert snapshot.consider(dataset_path="best.jsonl", score=1.0, dataset_hash="b")
    assert not snapshot.consider(dataset_path="worse.jsonl", score=3.0, dataset_hash="c")

