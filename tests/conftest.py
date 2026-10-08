"""Suite-wide setup. The network guard is installed before any test module is imported, so any
attempt to reach a non-loopback address fails the test (hard rule: no network calls besides
the local Ollama server)."""

from ratio import netguard

netguard.install()
