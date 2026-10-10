"""Suite-wide setup. The network guard is installed before any test module is imported, so any
attempt to reach a non-loopback address fails the test (hard rule: no network calls besides
the local Ollama server).

No test reads a real precedent corpus (or the OpenSearch index of one) unless it opts in:
RATIO_CORPUS_DB points at a file that does not exist, so Similar cases finds no corpus. A test that
needs one builds the SYNTHETIC fixture corpus, sets RATIO_CORPUS_DB to it, and reads the file. The
builder's work database and the uploaded collections are moved the same way."""

import os
import tempfile
import uuid
from pathlib import Path

from ratio import netguard

netguard.install()
# Assigned, never defaulted: a RATIO_CORPUS_DB exported in the shell must not point the suite at a real corpus.
_NOWHERE = Path(tempfile.gettempdir()) / f"ratio-tests-{uuid.uuid4().hex}"
os.environ["RATIO_CORPUS_DB"] = str(_NOWHERE / "no-corpus" / "precedents.db")
# Likewise the builder's work database and the uploaded collections: no test reads or writes the real ones.
os.environ["RATIO_BUILD_DB"] = str(_NOWHERE / "no-build" / "build.db")
os.environ["RATIO_COLLECTIONS_DIR"] = str(_NOWHERE / "no-collections")
