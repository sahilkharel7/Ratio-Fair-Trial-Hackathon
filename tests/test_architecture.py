"""Architecture rule from the spec: modules read only the case record, never raw text or files.

The modules may import the contract (schema, results, config, context, messages) and pure
libraries. They may not import the extraction layer, the store, the LLM client, Streamlit, or
anything that opens files or sockets.
"""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from ratio.paths import PACKAGE_DIR, REPO_ROOT

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


# --- Similar cases: the online builder and the offline runtime stay apart (hard rule 1) ------------

RUNTIME_FILES = sorted([*PACKAGE_DIR.rglob("*.py"), *(REPO_ROOT / "app").rglob("*.py")])
BUILDER_FILES = sorted((REPO_ROOT / "corpus_builder").glob("*.py"))
SEARCH_CLIENT = PACKAGE_DIR / "precedent_opensearch.py"


def _imports(path: Path) -> list[str]:
    return imported_names(ast.parse(path.read_text(encoding="utf-8")))


def _imports_any(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    return [name for name in _imports(path) if any(name == bad or name.startswith(bad + ".") for bad in prefixes)]


# The app may upload and read collections with the local model. It never constructs the cloud client, and the
# cloud SDK is never loaded (see the probe below).
APP_MAY_IMPORT = ("corpus_builder.collections", "corpus_builder.store")


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_the_app_never_imports_the_builder_or_the_cloud_model(path: Path):
    allowed = APP_MAY_IMPORT if (REPO_ROOT / "app") in path.parents else ()
    found = [
        name for name in _imports_any(path, ("google.genai", "google.generativeai", "corpus_builder"))
        if not any(name == ok or name.startswith(ok + ".") for ok in allowed)
    ]  # fmt: skip
    assert not found, path.name
    assert "GeminiLLM" not in path.read_text(encoding="utf-8"), path.name


def test_the_collections_module_is_installed_and_never_loads_the_cloud_sdk(tmp_path):
    """Run outside the repository, as `streamlit run app/main.py` imports it: from the installed package."""
    import sys

    probe = "import sys, corpus_builder.collections; print(any(m.startswith('google.genai') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert out.returncode == 0, f"corpus_builder is not installed; run `uv pip install -e .` ({out.stderr.strip().splitlines()[-1:]})"
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.parametrize("path", [*RUNTIME_FILES, *BUILDER_FILES], ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_only_the_search_backend_imports_opensearchpy(path: Path):
    if path != SEARCH_CLIENT:
        assert not _imports_any(path, ("opensearchpy",)), path.name


def test_the_search_backend_imports_opensearchpy_only_inside_functions():
    tree = ast.parse(SEARCH_CLIENT.read_text(encoding="utf-8"))
    top_level = imported_names(ast.Module(body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))], type_ignores=[]))
    assert not [name for name in top_level if name.startswith("opensearchpy")]
    assert any(name.startswith("opensearchpy") for name in imported_names(tree))  # the check sees the lazy imports


def test_similar_cases_obeys_the_modules_rules():
    path = PACKAGE_DIR / "precedents.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not _imports_any(path, FORBIDDEN_IMPORTS)
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & FORBIDDEN_CALLS


@pytest.mark.parametrize("path", BUILDER_FILES, ids=lambda p: p.name)
def test_the_builder_never_reads_cases_or_calls_the_case_model(path: Path):
    assert not _imports_any(path, ("ratio.store", "ratio.pipeline", "ratio.llm", "ratio.netguard.install")), path.name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    installs = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "install" and getattr(node.value, "id", "") == "netguard"]
    assert not installs, f"{path.name} installs the network guard"


def test_the_builder_rule_is_checking_real_files():
    assert any(path.name == "opensearch.py" for path in BUILDER_FILES)
    assert SEARCH_CLIENT in RUNTIME_FILES


def test_the_precedent_corpus_is_never_committed():
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(["git", "check-ignore", "-q", "data/corpus/x"], cwd=REPO_ROOT, check=False)
    assert ignored.returncode == 0, "data/corpus/ must be git-ignored: it holds real public documents"
