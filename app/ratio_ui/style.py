"""Stylesheet and the SYNTHETIC banner. System fonts only: the app never fetches anything."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

DOCUMENT_CSS = """
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

# A single stylesheet owns the application tokens and the Streamlit adapter.
# System fonts keep the research workspace fully usable offline.
CSS = Path(__file__).with_name("theme.css").read_text(encoding="utf-8") + DOCUMENT_CSS


def inject() -> None:
    st.html(f"<style>{CSS}</style>")


def banner(text: str) -> None:
    st.html(f'<div class="ratio-banner">{html.escape(text)}</div>')


def eyebrow(text: str) -> None:
    st.html(f'<div class="ratio-eyebrow">{html.escape(text)}</div>')


def section(title: str, caption: str = "") -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def panel(key: str, *, height: int | str = "content"):
    """Native container with a stable styling hook, rather than framework classes."""
    return st.container(border=True, height=height, key=f"ratio-panel-{key}")


def metric(label: str, value: str | int, note: str, tone: str = "neutral") -> None:
    """Compact summary with an explicit label; colour never carries meaning alone."""
    if tone not in {"neutral", "review", "positive"}:
        tone = "neutral"
    st.html(
        f'<div class="ratio-metric ratio-metric--{tone}">'
        f'<div class="ratio-metric-label">{html.escape(label)}</div>'
        f'<div class="ratio-metric-value">{html.escape(str(value))}</div>'
        f'<div class="ratio-metric-note">{html.escape(note)}</div></div>'
    )


def masthead() -> None:
    st.html(
        '<div class="ratio-masthead"><span>TRIAL REVIEW WORKSPACE</span>'
        '<span class="ratio-local"><i aria-hidden="true"></i> Offline · On this computer</span></div>'
    )


def sidebar(active: str = "") -> None:
    """Persistent navigation. Native page links retain active and keyboard states."""
    with st.sidebar:
        st.html(
            '<div class="ratio-brand"><span class="ratio-monogram" aria-hidden="true">r.</span>'
            '<div><span class="ratio-wordmark">ratio</span><span class="ratio-brand-caption">FAIR TRIAL ANALYSIS</span></div></div>'
        )
        eyebrow("Review workspace")
        links = (
            ("", "case", "Case overview"),
            ("coverage", "coverage", "Rights coverage"),
            ("timeline", "timeline", "Procedural timeline"),
            ("reuse", "reuse", "Reasoning comparison"),
            ("judges", "judges", "Judicial history"),
        )
        for route, name, title in links:
            with st.container(key=f"ratio-nav-{name}"):
                st.page_link(f"views/{name}.py", label=title)
            if route == active:
                st.html(f'<style>.st-key-ratio-nav-{name} a {{background: var(--ratio-accent-soft) !important; border-left-color: var(--ratio-accent) !important; color: var(--ratio-accent) !important;}}</style>')
        st.html(
            '<div class="ratio-sidebar-note"><div class="ratio-eyebrow">BUILT FOR THE RECORD</div>'
            '<p>From observation to evidence.<br>Every finding, back to its source.</p>'
            '<div class="ratio-sidebar-rule"></div><p>ICCPR Articles 9 &amp; 14</p>'
            '<small>Analysis supports your review.<br>Legal conclusions remain yours.</small></div>'
        )
