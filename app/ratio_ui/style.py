"""Stylesheet and the SYNTHETIC banner. System fonts only: the app never fetches anything."""

from __future__ import annotations

import html

import streamlit as st

CSS = """
.ratio-doc { white-space: pre-wrap; font-family: Georgia, "Times New Roman", serif; font-size: 0.95rem; line-height: 1.65; }
.ratio-doc mark { color: inherit; padding: 0 1px; border-radius: 2px; }
mark.ratio-span { background: rgba(255, 193, 7, 0.5); box-shadow: 0 0 0 2px rgba(230, 150, 0, 0.9); }
mark.ratio-verbatim { background: rgba(255, 193, 7, 0.42); }
mark.ratio-paraphrase { background: rgba(66, 133, 244, 0.28); }
mark.ratio-charge { background: rgba(140, 140, 140, 0.25); text-decoration: underline dotted; }
.ratio-excluded { opacity: 0.45; }
.ratio-tag { font-family: system-ui, sans-serif; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.04em;
  border: 1px solid currentColor; border-radius: 3px; padding: 0 4px; margin: 0 4px; opacity: 0.75; white-space: nowrap; }
sup.ratio-label { font-family: system-ui, sans-serif; font-size: 0.66rem; font-weight: 700; margin-left: 1px; }
.ratio-banner { background: #fff4e0; color: #5c3b00; border: 1px solid #e8a33d; border-radius: 6px; padding: 6px 12px;
  font-family: system-ui, sans-serif; font-size: 0.85rem; font-weight: 600; }
.ratio-quote { font-family: Georgia, "Times New Roman", serif; font-size: 0.93rem; line-height: 1.5; }
.ratio-quote .ratio-where { font-family: system-ui, sans-serif; font-size: 0.75rem; opacity: 0.7; display: block; }
.ratio-legend mark { padding: 0 6px; margin-right: 10px; }
"""


def inject() -> None:
    st.html(f"<style>{CSS}</style>")


def banner(text: str) -> None:
    st.html(f'<div class="ratio-banner">{html.escape(text)}</div>')
