"""Stylesheet and the SYNTHETIC banner. Bundled fonts only: the app never fetches anything.

Colours live in .streamlit/config.toml. The few set here sit on their own background (marks, the
banner), so they read the same in the light and the dark theme; each text colour meets WCAG AA, and
no mark is told apart by colour alone (each has its own underline).
"""

from __future__ import annotations

import html

import streamlit as st

CSS = """
.ratio-doc { white-space: pre-wrap; font-family: "Source Serif", Georgia, serif; font-size: 1rem; line-height: 1.7; }
mark.ratio-span, mark.ratio-verbatim, mark.ratio-paraphrase, mark.ratio-charge { color: #0F172A; padding: 0 2px; border-radius: 2px; }
mark.ratio-span { background: #FDE68A; box-shadow: 0 0 0 2px #B45309; }
mark.ratio-verbatim { background: #FDE68A; border-bottom: 2px solid #B45309; }
mark.ratio-paraphrase { background: #DBEAFE; border-bottom: 2px dashed #1D4ED8; }
mark.ratio-charge { background: #E2E8F0; border-bottom: 2px dotted #475569; }
.ratio-excluded { opacity: 0.62; }
.ratio-tag { font-family: "Source Sans", system-ui, sans-serif; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
  border: 1px solid currentColor; border-radius: 2px; padding: 0 4px; margin: 0 4px; white-space: nowrap; }
sup.ratio-label { font-family: "Source Sans", system-ui, sans-serif; font-size: 0.7rem; font-weight: 700; margin-left: 1px; }
.ratio-banner { background: #FFF7ED; color: #7C2D12; border: 1px solid #FDBA74; border-left: 4px solid #C2410C; border-radius: 2px;
  padding: 8px 14px; font-family: "Source Sans", system-ui, sans-serif; font-size: 0.875rem; font-weight: 600; }
.ratio-quote { font-family: "Source Serif", Georgia, serif; font-size: 0.98rem; line-height: 1.55;
  border-left: 3px solid #94A3B8; padding: 1px 0 1px 14px; }
.ratio-quote .ratio-where { font-family: "Source Sans", system-ui, sans-serif; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.01em;
  opacity: 0.8; display: block; margin-bottom: 2px; }
.ratio-legend mark { padding: 0 6px; margin-right: 10px; }
.ratio-offline { font-size: 0.8rem; line-height: 1.45; opacity: 0.8; }
:is(button, a, summary, [role="tab"], [role="button"], input, textarea, select):focus-visible {
  outline: 2px solid #2563EB !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}
@media print {
  [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"], .stButton { display: none !important; }
  .ratio-doc, .ratio-quote { font-size: 11pt; }
}
"""


def inject() -> None:
    st.html(f"<style>{CSS}</style>")


def banner(text: str) -> None:
    st.html(f'<div class="ratio-banner">{html.escape(text)}</div>')


def offline_note(text: str) -> None:
    """A short line under the page list that says where the record and the model run."""
    st.html(f'<div class="ratio-offline">{html.escape(text)}</div>')
