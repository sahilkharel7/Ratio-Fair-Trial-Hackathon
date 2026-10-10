"""Similar cases: past cases (Views of the UN Human Rights Committee, opinions of the UN Working Group on
Arbitrary Detention, TrialWatch reports, and the collections uploaded on this computer) that share this
case's fact patterns. Each shared
pattern shows this case's passage beside the past case's, both opening their exact, checked source.
A reading aid, never a finding and never a prediction: whether a precedent applies is for the
reviewing lawyer."""

from __future__ import annotations

import logging

import streamlit as st

from ratio.display import md_escape
from ratio.embeddings import EmbeddingModelMissing
from ratio.messages import render
from ratio import precedents
from ratio.precedent_schema import CaseProfile, CorpusMeta, PrecedentIndex, PrecedentLink, SharedFacet
from ratio.precedents import WordingMatch
from ratio.schema import Evidence
from ratio_ui import session, viewer, widgets
from ratio_ui.session import SimilarCases

_log = logging.getLogger(__name__)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _no_corpus(reason: str) -> None:
    st.info(md_escape(notes["similar_no_corpus"]))
    st.caption(md_escape(reason))


def _profile(found: CaseProfile) -> None:
    """The fact patterns the links rest on, and where each came from in this case."""
    st.header("Fact patterns in this case")
    if not found.facets:
        st.caption(md_escape(notes["similar_no_facets"]))
        return
    lines = (
        f"- {md_escape(cfg.fact_patterns.facet(facet.facet_id).label)} ({md_escape(widgets.label('facet_from_' + facet.origin))})"
        for facet in found.facets
    )
    st.markdown("\n".join(lines))


def _score(found: PrecedentLink) -> str:
    patterns = _count(found.score.shared, "fact pattern")
    cosine = found.score.mean_pair_cosine
    if cosine is None:
        return render(cfg.messages, "similar_score_no_cosine", patterns=patterns)
    return render(cfg.messages, "similar_score", patterns=patterns, cosine=f"{cosine:.2f}")


def _shared(found: PrecedentLink, shared: SharedFacet) -> None:
    """One shared fact pattern: this case's passage on the left, the past case's on the right."""
    doc = found.precedent
    key = f"similar-{doc.id}-{shared.facet_id}"
    st.markdown(f"**{md_escape(shared.label)}**")
    case_column, precedent_column = st.columns(2, gap="medium")
    with case_column:
        st.caption(md_escape(widgets.label("similar_this_case")))
        widgets.narrow_evidence(loaded.record, shared.case_evidence, key=f"{key}-case", heading=shared.label)
    with precedent_column:
        st.caption(md_escape(doc.body))
        widgets.precedent_evidence(doc, shared, key=f"{key}-precedent")


def _card(found: PrecedentLink) -> None:
    doc = found.precedent
    with st.container(border=True):
        st.subheader(md_escape(doc.title))
        source, origin = widgets.label(f"precedent_{doc.kind}"), viewer.precedent_origin(doc)
        st.markdown(f":blue-badge[{md_escape(source)}] :gray-badge[{md_escape(origin)}]")
        details = (doc.body, doc.symbol, doc.state, str(doc.year) if doc.year else None)
        st.caption(md_escape(" · ".join(part for part in details if part)))
        st.caption(md_escape(_score(found)))
        for shared in found.shared:
            _shared(found, shared)


def _corpus(meta: CorpusMeta) -> None:
    """Where the precedents come from, on what terms, and how the corpus was built."""
    with st.expander(notes["similar_corpus_heading"]):
        st.caption(md_escape(notes["similar_attribution"]))
        rows = [
            {"Source": s.name, "Documents": s.documents, "Attribution": s.attribution, "Terms of use": s.terms_url, "Terms checked": s.terms_checked}
            for s in meta.sources
        ]  # a table shows the terms address as plain text, never as a link out of the app
        st.dataframe(rows, hide_index=True)
        built = render(
            cfg.messages, "similar_built", date=meta.built_at[:10], documents=_count(meta.documents, "document"),
            facets=meta.facets, model=meta.extraction_model or "not recorded", dropped=meta.dropped_quotes,
        )  # fmt: skip
        st.caption(md_escape(built))


def _failed(exc: Exception) -> None:
    """A message instead of a traceback; the reason goes to the log."""
    if isinstance(exc, EmbeddingModelMissing):
        st.error(md_escape(str(exc)))  # it says what to download
        return
    _log.warning("Similar cases could not be shown", exc_info=exc)
    st.error(md_escape(notes["similar_failed"]))


def _computing() -> None:
    """Still computing after the page's wait: the links keep computing, and a click looks again."""
    st.caption(md_escape(notes["similar_computing"]))
    st.button(widgets.label("similar_check_again"), key="similar_check_again")  # a click runs the page again


def _results(found: SimilarCases) -> None:
    _profile(found.profile or CaseProfile(case_id=loaded.record.case_id))
    st.header("Past cases that share them")
    links = found.links or ()
    if not links:
        st.caption(md_escape(render(cfg.messages, "similar_none", min_shared=session.precedent_settings().min_shared)))
    for link in links:
        _card(link)


def _wording_match(match: WordingMatch, number: int) -> None:
    doc = match.precedent
    with st.container(border=True):
        st.subheader(md_escape(doc.title))
        source, origin = widgets.label(f"precedent_{doc.kind}"), viewer.precedent_origin(doc)
        st.markdown(f":blue-badge[{md_escape(source)}] :gray-badge[{md_escape(origin)}]")
        paragraphs = _count(match.support, "paragraph")
        st.caption(md_escape(f"{doc.body} · closeness of the best pair {match.cosine:.2f} · {paragraphs} of this case read close to it"))
        case_column, precedent_column = st.columns(2, gap="medium")
        with case_column:
            st.caption(md_escape(widgets.label("similar_this_case")))
            widgets.narrow_evidence(loaded.record, [Evidence(role="mention", span=match.case_passage)], key=f"wording-{number}-case", heading="Similar wording")
        with precedent_column:
            st.caption(md_escape(doc.body))
            st.html(widgets.precedent_quote_html(widgets.label("library_passage"), match.precedent_passage, clip=False))
            if st.button("View source", key=f"wording-{number}-precedent", help="Opens the passage, checked against its document"):
                viewer.show_precedent(match.precedent_passage.span, doc, "Similar wording")


def _wording(index: PrecedentIndex) -> None:
    """Past cases whose paragraphs read closest to this case's own (computed once per case and corpus)."""
    st.header(notes["similar_wording_heading"])
    st.caption(md_escape(notes["similar_wording_intro"]))
    key = f"similar-wording/{loaded.record.case_id}/{index.meta().built_at}"
    if key not in st.session_state:
        with st.spinner("Searching the library with this case's paragraphs"):
            st.session_state[key] = precedents.similar_wording(loaded.record, index, session.embedder())
    matches = st.session_state[key]
    if not matches:
        st.caption(md_escape(notes["similar_wording_none"]))
    for number, match in enumerate(matches):
        _wording_match(match, number)


def _similar(index: PrecedentIndex, status: str) -> None:
    st.caption(md_escape(render(cfg.messages, "similar_status", status=status)))
    with st.spinner("Comparing this case with the past cases"):  # computed once, then shared with the Case page
        found = session.similar_cases(loaded, timeout=session.SIMILAR_PAGE_WAIT_SECONDS)
    if found is None:
        _computing()
    else:
        _results(found)
    try:
        _wording(index)
    except Exception as exc:  # noqa: BLE001 - this section is optional: the rest of the page stays
        _log.warning("Similar wording could not be shown", exc_info=exc)
    _corpus(index.meta())


loaded = widgets.require_case()
cfg = session.config()
notes = cfg.messages.notes
widgets.header(loaded, "Similar cases", "Past cases that share this case's fact patterns")
st.markdown(md_escape(notes["similar_intro"]))
st.caption(md_escape(notes["similar_caveat"]))
try:
    index, status = session.precedent_index()
    if index is None:
        _no_corpus(status)
    else:
        _similar(index, status)
except Exception as exc:  # noqa: BLE001 - shown as a message, never a traceback
    _failed(exc)
