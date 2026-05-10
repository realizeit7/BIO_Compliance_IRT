"""
Mechanically enforce CRITICAL ENGINEERING RULE 2:
the /evaluator/ layer must never import an LLM SDK or the /harness/ layer.

Runs as a unit test. Execute via:
    python3 -m unittest tests.test_evaluator_isolation
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = REPO_ROOT / "evaluator"

FORBIDDEN_PREFIXES = (
    "openai",
    "groq",
    "anthropic",
    "harness",
    "agents",        # /agents/ is allowed to import /harness/, so /evaluator/ must avoid it too
    "arena.runner",  # arena.runner imports /harness/ transitively
)


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
    return found


class EvaluatorIsolationTest(unittest.TestCase):
    def test_no_forbidden_imports(self) -> None:
        offenders: list[tuple[str, str]] = []
        for py in EVALUATOR_DIR.rglob("*.py"):
            for mod in _imports_in_file(py):
                for prefix in FORBIDDEN_PREFIXES:
                    if mod == prefix or mod.startswith(prefix + "."):
                        offenders.append((str(py.relative_to(REPO_ROOT)), mod))
        self.assertFalse(
            offenders,
            f"/evaluator/ must not import LLM/harness modules. Offenders: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
