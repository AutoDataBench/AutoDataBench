import json

from bench_core.artifacts import RunArtifacts


def test_only_best_dataset_and_checkpoint_are_retained(tmp_path) -> None:
    run = RunArtifacts.create("stub", tmp_path)
    first = run.workspace / "first.jsonl"
    first.write_text('{"id": 1}\n')
    sequence, digest, checkpoint = run.start_eval(str(first))
    (checkpoint / "weights.bin").write_bytes(b"first")
    saved = run.finish_eval(
        sequence=sequence,
        dataset_path=str(first),
        dataset_hash=digest,
        checkpoint=checkpoint,
        result={"val_scores": [0.8], "n_samples_trained": 1},
        mean_score=0.8,
        best_updated=True,
        quota={},
    )
    assert saved == str(run.artifacts_dir / "best_checkpoint")
    assert (run.artifacts_dir / "best_dataset.jsonl").read_text() == first.read_text()

    worse = run.workspace / "worse.jsonl"
    worse.write_text('{"id": 2}\n')
    sequence, digest, checkpoint = run.start_eval(str(worse))
    (checkpoint / "weights.bin").write_bytes(b"worse")
    run.finish_eval(
        sequence=sequence,
        dataset_path=str(worse),
        dataset_hash=digest,
        checkpoint=checkpoint,
        result={"val_scores": [0.2], "n_samples_trained": 1},
        mean_score=0.2,
        best_updated=False,
        quota={},
    )
    assert (run.artifacts_dir / "best_checkpoint" / "weights.bin").read_bytes() == b"first"
    assert not checkpoint.exists()
    assert len(json.loads((run.artifacts_dir / "evaluations.json").read_text())) == 2


def test_final_dataset_and_results_are_durable(tmp_path) -> None:
    run = RunArtifacts.create("stub", tmp_path)
    dataset = run.workspace / "final.jsonl"
    dataset.write_text('{"id": 3}\n')
    saved, digest = run.save_final_dataset(str(dataset))
    run.save_results({"final_dataset": saved, "final_dataset_hash": digest})

    assert (run.artifacts_dir / "final_dataset.jsonl").is_file()
    assert json.loads((run.artifacts_dir / "results.json").read_text())["final_dataset_hash"] == digest
    assert (run.artifacts_dir / "trajectory.jsonl").is_file()
