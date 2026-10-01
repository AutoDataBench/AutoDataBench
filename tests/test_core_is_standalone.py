import ast
from pathlib import Path


def test_core_does_not_import_runtime_or_ml_packages() -> None:
    forbidden = {"inspect_ai", "torch", "transformers", "vllm", "httpx"}

    for path in Path("bench_core").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden), f"forbidden import in {path}"

