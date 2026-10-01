import json

import pytest

from bench_core.quota import QuotaConfig
from bench_helpers import bridge
from bench_inspect.bridge import BridgeHandler, extract_bridge_seq
from bench_inspect.state import current, initialize, resolve_workspace_path

from test_runtime import Model, Training, Validator


def test_bridge_replays_cached_calls_in_order(tmp_path, capsys) -> None:
    bridge.BRIDGE_DIR = str(tmp_path)
    bridge._call_seq = 0

    with pytest.raises(SystemExit) as exit_info:
        bridge.small_model_embed(["first"])
    assert exit_info.value.code == 99
    assert extract_bridge_seq(capsys.readouterr().out) == 1
    request = json.loads((tmp_path / "req_1.json").read_text())
    assert request["method"] == "small_model_embed"
    (tmp_path / "resp_1.json").write_text(
        json.dumps({"result": {"embeddings": [[1.0]]}})
    )

    bridge._call_seq = 0
    assert bridge.small_model_embed(["first"])["embeddings"] == [[1.0]]
    with pytest.raises(SystemExit):
        bridge.validate_format("/workspace/candidate.jsonl")
    assert json.loads((tmp_path / "req_2.json").read_text())["seq"] == 2


@pytest.mark.anyio
async def test_bridge_uses_shared_quota_and_best_snapshot(tmp_path) -> None:
    initialize(
        QuotaConfig.from_dict(
            {
                "eval_calls": {"limit": 2},
                "train_samples": {"limit": 10},
            }
        ),
        workspace=tmp_path,
    )
    handler = BridgeHandler(Model(), Training(), Validator())
    response = await handler.handle(
        {
            "method": "train_and_eval",
            "params": {"dataset_path": "/workspace/candidate.jsonl", "n_seeds": 1},
            "seq": 1,
        }
    )

    assert "error" not in response
    assert current().quota.spent("eval_calls") == 1
    assert current().quota.spent("train_samples") == 4
    assert current().best.dataset_path == str(tmp_path / "candidate.jsonl")
    assert resolve_workspace_path("/workspace/train.jsonl") == str(tmp_path / "train.jsonl")
