"""Arguments the model missed: a note sentence in which a party argues "that ...", and whose whole
subject, right before the verb, is that party ("Defence counsel argued that ...", "In closing,
Ms. Quint argued that ..."). A named speaker counts only if the notes name them as counsel ("her
lawyer, Mara Quint"). Negated, reported or passive sentences, list announcements ("argued two
points.") and other speakers (a witness, the judge) are skipped. Each argument is the exact note
sentence and is marked for review, since no model read it as an argument.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ratio.schema import Argument, CaseRecord, Party, stable_id

# A verb that introduces what the party said.
_CLAIM = re.compile(r"\b(?:argued|contended|submitted|maintained|insisted|claimed|asserted|objected|complained)\s+that\b", re.IGNORECASE)
_TITLE = r"(?:Ms|Mr|Mrs|Dr|Prof|Me|Adv)\."
_NAME_WORD = r"[A-Z][\w'-]+"
# Named counsel within one sentence: "her lawyer, Mara Quint", "Advocate Lena Reyl", "Prosecutor Edda Marn".
_COUNSEL = re.compile(
    rf"\b(?i:(?P<role>lawyer|counsel|advocate|prosecutor))\b,?[ \t]+(?:{_TITLE}[ \t]+)?(?P<name>{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}})"
)
_NOT_A_NAME = frozenset(
    {"Ms", "Mr", "Mrs", "Dr", "Prof", "Me", "Adv", "The", "A", "An", "In", "On", "At", "For", "Judge", "Justice",
     "Inspector", "Officer", "Detective", "Sergeant", "Captain", "Witness", "Court", "President", "Presiding"}
)  # fmt: skip
_PROSECUTION_COUNSEL = re.compile(r"\b(?:prosecut\w*|state|crown)\W*$", re.IGNORECASE)  # "prosecution counsel X"
_OTHER_COUNSEL = re.compile(r"\b(?:victim|civil party|complainant|witness)\w*'?s?\W*$", re.IGNORECASE)  # "the victim's lawyer X"
_DEFENCE_SUBJECT = re.compile(r"(?:the )?(?:defen[cs]e(?: counsel| lawyer)?|counsel for the defen[cs]e)", re.IGNORECASE)
_PROSECUTION_SUBJECT = re.compile(r"(?:the )?(?:(?:public |state )?prosecutor|prosecution)", re.IGNORECASE)
_NAMED_SUBJECT = re.compile(rf"(?:{_TITLE} )?(?:{_NAME_WORD} )?(?P<surname>{_NAME_WORD})")


def counsel_roster(record: CaseRecord) -> dict[str, Party]:
    """The party of each counsel the notes name, by surname; a surname named for both sides is left out."""
    roster: dict[str, Party] = {}
    clashes: set[str] = set()
    for obs in record.observations:
        for found in _COUNSEL.finditer(obs.text):
            before, words = obs.text[: found.start()], found.group("name").split()
            if words[0] in _NOT_A_NAME or words[-1] in _NOT_A_NAME or _OTHER_COUNSEL.search(before):
                continue
            prosecution = found.group("role").lower() == "prosecutor" or _PROSECUTION_COUNSEL.search(before)
            party: Party = "prosecution" if prosecution else "defense"
            if roster.get(words[-1], party) != party:
                clashes.add(words[-1])
            roster[words[-1]] = party
    return {surname: party for surname, party in roster.items() if surname not in clashes}


def _speaker(subject: str, roster: dict[str, Party]) -> Party | None:
    """The party that is the whole subject of the verb, after an opener such as "In closing,"."""
    clause = " ".join(subject.rsplit(",", 1)[-1].split())
    if _DEFENCE_SUBJECT.fullmatch(clause):
        return "defense"
    if _PROSECUTION_SUBJECT.fullmatch(clause):
        return "prosecution"
    named = _NAMED_SUBJECT.fullmatch(clause)
    return roster.get(named.group("surname")) if named else None


def missed_arguments(base: CaseRecord, kept: Sequence[Argument]) -> list[Argument]:
    """Arguments in the record's documents that the model did not return (see the module docstring)."""
    roster = counsel_roster(base)
    documents = {doc.id for doc in base.documents}
    added = []
    for obs in base.observations:
        claim = _CLAIM.search(obs.text)
        taken = any(a.span.doc_id == obs.span.doc_id and a.span.start < obs.span.end and obs.span.start < a.span.end for a in kept)
        party = _speaker(obs.text[: claim.start()], roster) if claim and not taken and obs.span.doc_id in documents else None
        if party is not None:
            added.append(
                Argument(
                    id=stable_id(obs.span.doc_id, obs.span.start, party, "argument"),
                    party=party,
                    text=obs.text,
                    hearing_date=obs.hearing_date,
                    span=obs.span,
                    quote_span=obs.span,
                    needs_review=True,
                )
            )
    return added
