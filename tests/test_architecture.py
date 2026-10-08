"""Architecture rule from the spec: modules read only the case record, never raw text or files.

The modules may import the contract (schema, results, config, context, messages) and pure
libraries. They may not import the extraction layer, the store, the LLM client, Streamlit, or
anything that opens files or sockets.
"""

import ast
from pathlib import Path

import pytest

from ratio.paths import PACKAGE_DIR

FORBIDDEN_IMPORTS = (
    "ratio.extraction",
    "ratio.store",
    "ratio.llm",
    "ratio.pipeline",
    "ratio.gold",
    "ratio.paths",
    "streamlit",
    "sqlite3",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "pathlib",
    "os",
    "io",
)
FORBIDDEN_CALLS = frozenset({"open", "exec", "eval", "__import__", "import_module"})

MODULE_FILES = sorted((PACKAGE_DIR / "modules").glob("*.py"))


def imported_names(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative imports would bypass the check
                names.append("." * node.level + (node.module or ""))
            elif node.module:
                names.append(node.module)
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_modules_only_import_the_contract_and_pure_libraries(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name in imported_names(tree):
        assert not name.startswith("."), f"{path.name} uses a relative import"
        assert not any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_IMPORTS), (
            f"{path.name} imports {name}"
        )


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_modules_never_open_files_or_evaluate_code(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & FORBIDDEN_CALLS, f"{path.name} calls {sorted(calls & FORBIDDEN_CALLS)}"


def test_the_rule_is_checking_real_files():
    assert any(path.name == "__init__.py" for path in MODULE_FILES)
