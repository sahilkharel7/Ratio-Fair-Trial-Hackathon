"""Judge registry for the Judicial History Tracker.

Each case names its judges in the coded rulings (rulings.yaml). Names are transliterated
(unidecode), lower-cased and stripped of leading titles, then compared with rapidfuzz
token_sort_ratio, and only ever within one court and one kind of data (synthetic or public).
At or above the match threshold the names are one judge. A name waits on the manual confirmation
list, and its rulings count nowhere until a person decides (alias_decisions.yaml), when it is:
close to a registered name but below the threshold; a shorter or longer form of one (a surname
alone, an added role); written with initials only; only a title; or seen at the same court years
apart from the judge's other cases. The result does not depend on the order of the cases.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from rapidfuzz import fuzz
from unidecode import unidecode

from ratio.config import JudgeSettings
from ratio.results import AliasCandidate, DataNote
from ratio.schema import AliasDecision, CaseRecord, stable_id

_WORD = re.compile(r"[a-z]+")
_APOSTROPHE = re.compile(r"['`]")  # O'Neill is one word, not an initial and a surname
_DAYS_PER_YEAR = 365.25


def normalise_name(raw: str, titles: Iterable[str]) -> str:
    """'Hon. Iléna Várda' -> 'ilena varda'. Only leading titles are stripped ('Mary Judge' keeps both
    words); a name made of titles only ('The Presiding Judge') has no key at all."""
    tokens = _words(raw)
    title_words = set(titles)
    start = 0
    while start < len(tokens) and tokens[start] in title_words:
        start += 1
    return " ".join(tokens[start:])


def _words(raw: str) -> list[str]:
    return _WORD.findall(_APOSTROPHE.sub("", unidecode(raw).lower()))


def key_text(text: str) -> str:
    """Comparison key for court and charge names: case and spacing do not matter."""
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class RegisteredJudge:
    judge_id: str
    display_name: str
    court: str
    synthetic: bool
    variants: tuple[str, ...]
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    judges: tuple[RegisteredJudge, ...]
    pending: tuple[AliasCandidate, ...]
    assignment: Mapping[tuple[str, str], str]  # (case id, name as written) -> judge id
    notes: tuple[DataNote, ...] = ()

    def judge_of(self, case_id: str, raw_name: str) -> str | None:
        return self.assignment.get((case_id, raw_name))


@dataclass(frozen=True)
class _Occurrence:
    """One name as written in one case's rulings."""

    case_id: str
    raw_name: str
    court: str
    synthetic: bool
    dates: tuple[dt.date, ...]
    name_key: str
    decided_as: str | None = None  # a person decided this is the judge registered under that name
    apart: bool = False  # a person decided it is not the judge it resembles

    @property
    def place(self) -> tuple[bool, str]:
        return self.synthetic, key_text(self.court)


@dataclass(frozen=True)
class _Cluster:
    """Occurrences treated as one judge."""

    occurrences: tuple[_Occurrence, ...]

    @property
    def name_keys(self) -> frozenset[str]:
        return frozenset(o.name_key for o in self.occurrences)

    @property
    def cases(self) -> int:
        return len({o.case_id for o in self.occurrences})

    @property
    def display_name(self) -> str:
        """The most common spelling, preferring names matched without a decision (a confirmed 'I. Varda'
        does not rename the judge) and, on a tie, the shortest (fewest titles)."""
        matched = [o for o in self.occurrences if o.decided_as is None] or self.occurrences
        counts = Counter(o.raw_name for o in matched)
        return min(counts, key=lambda name: (-counts[name], len(name), name))


def _decision_notes(found: set[tuple[str, str]], decisions: Sequence[AliasDecision], kinds: Mapping[str, bool]) -> tuple[dict, list[DataNote]]:
    chosen: dict[tuple[str, str], AliasDecision] = {}
    conflicting: set[tuple[str, str]] = set()
    notes = []
    for decision in decisions:
        key = (decision.case_id, decision.name)
        if decision.case_id not in kinds:
            text = f"alias_decisions.yaml: case {decision.case_id} is not stored, so its decision was not used"
            notes.append(DataNote(text=text, synthetic=decision.synthetic))
        elif decision.synthetic is not None and decision.synthetic != kinds[decision.case_id]:
            recorded, actual = ("synthetic", "public") if decision.synthetic else ("public", "synthetic")
            text = f"alias_decisions.yaml: the decision on case {decision.case_id} is in a {recorded} file but the case is {actual}, so it was not used"
            notes.append(DataNote(text=text, synthetic=kinds[decision.case_id]))
            continue
        elif key not in found:
            text = f"alias_decisions.yaml: no ruling in case {decision.case_id} names {decision.name!r}, so the decision was not used"
            notes.append(DataNote(text=text, synthetic=kinds[decision.case_id]))
        elif key in chosen and chosen[key] != decision:
            conflicting.add(key)
        chosen.setdefault(key, decision)
    for case_id, name in sorted(conflicting):
        text = f"alias_decisions.yaml: conflicting decisions for {name!r} in case {case_id}, so neither was used"
        notes.append(DataNote(text=text, synthetic=kinds[case_id]))
    return {key: value for key, value in chosen.items() if key in found and key not in conflicting}, notes


def _occurrences(records: Sequence[CaseRecord], decisions: Sequence[AliasDecision], titles: tuple[str, ...]) -> tuple[list[_Occurrence], list[DataNote]]:
    by_case = {record.case_id: record for record in records}
    found = {(record.case_id, ruling.judge_name) for record in records for ruling in record.rulings}
    decided, notes = _decision_notes(found, decisions, {case_id: r.meta.synthetic for case_id, r in by_case.items()})
    occurrences = []
    for case_id, name in sorted(found):
        record = by_case[case_id]
        decision = decided.get((case_id, name))
        same_as = decision.same_as if decision is not None else None
        apart = decision is not None and same_as is None
        key = normalise_name(same_as or name, titles)
        if apart and not key:
            key = " ".join(_words(name))  # 'Judge Lord', decided to be a judge of its own: keep every word
        occurrences.append(
            _Occurrence(
                case_id=case_id,
                raw_name=name,
                court=record.meta.court,
                synthetic=record.meta.synthetic,
                dates=tuple(sorted({r.date for r in record.rulings if r.judge_name == name and r.date is not None})),
                name_key=key,
                decided_as=same_as,
                apart=apart,
            )
        )
    return occurrences, notes


def _core(name_key: str, titles: Iterable[str]) -> str:
    """The name without title or role words anywhere: 'i varda presiding' -> 'i varda'."""
    title_words = set(titles)
    return " ".join(token for token in name_key.split() if token not in title_words)


def _initials_only(name_key: str, titles: Iterable[str]) -> bool:
    """'i varda', 'i m varda' or 'judge i varda presiding': the given names reduced to initials.
    'ilena m varda' is a full name with a middle initial."""
    tokens = _core(name_key, titles).split()
    return any(len(t) == 1 for t in tokens) and sum(len(t) > 1 for t in tokens) <= 1


def _initials_conflict(a: str, b: str) -> bool:
    """'ilena m varda' and 'ilena p varda' are two people: one side's initial matches none of the other
    side's own words while that side has words of its own. 'ilena m varda' and 'ilena mira varda' fit."""
    words_a, words_b = a.split(), b.split()
    only_a = [w for w in words_a if w not in words_b]
    only_b = [w for w in words_b if w not in words_a]

    def clash(mine: list[str], theirs: list[str]) -> bool:
        return any(len(w) == 1 and theirs and not any(t[0] == w for t in theirs) for w in mine)

    return clash(only_a, only_b) or clash(only_b, only_a)


def _conflicting(a: Iterable[str], b: Iterable[str], titles: Iterable[str]) -> bool:
    return any(_initials_conflict(_core(x, titles), _core(y, titles)) for x in a for y in b)


def _same_word(word: str, candidates: Sequence[str], floor: float) -> str | None:
    """The candidate that is this word, or close to it ('vardah' and 'varda')."""
    if word in candidates:
        return word
    close = [c for c in candidates if len(c) > 1 and fuzz.ratio(word, c) >= floor]
    return max(close, key=lambda c: (fuzz.ratio(word, c), c)) if close else None


def _initials_fit(initials_name: str, full_names: Iterable[str], floor: float = 100.0) -> bool:
    """'i varda' fits 'ilena varda' (and, with a floor below 100, 'i vardah' does too); 'p varda' does not.
    An extra middle initial fits."""
    words = [t for t in initials_name.split() if len(t) > 1]
    letters = [t for t in initials_name.split() if len(t) == 1]
    for full in full_names:
        remaining = full.split()
        matched = [_same_word(word, remaining, floor) for word in words]
        if None in matched or len(set(matched)) != len(matched):
            continue
        for word in matched:
            remaining.remove(word)
        fits = True
        for letter in letters:
            match = next((t for t in remaining if t[0] == letter), None)
            if match is not None:
                remaining.remove(match)
            elif remaining:
                fits = False
        if fits:
            return True
    return False


def _score(a: Iterable[str], b: Iterable[str]) -> float:
    return max(fuzz.token_sort_ratio(x, y) for x in a for y in b)


def _contained(a: Iterable[str], b: Iterable[str], titles: tuple[str, ...]) -> bool:
    """One name is the other plus or minus words ('varda' and 'ilena varda'; 'ilena varda presiding')."""
    title_words = set(titles)
    cores_a = [frozenset(key.split()) - title_words for key in a]
    cores_b = [frozenset(key.split()) - title_words for key in b]
    return any(x and y and (x <= y or y <= x) for x in cores_a for y in cores_b)


def _clusters(occurrences: Sequence[_Occurrence], settings: JudgeSettings) -> list[_Cluster]:
    """Names at or above the threshold become one cluster, unless any two of their names carry different
    initials (keys are visited in sorted order, so the result does not depend on the order of the cases)."""
    keys = sorted({o.name_key for o in occurrences})
    parent = {key: key for key in keys}
    members = {key: [key] for key in keys}

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ra, rb = root(a), root(b)
            if ra == rb or fuzz.token_sort_ratio(a, b) < settings.name_match_threshold:
                continue
            if _conflicting(members[ra], members[rb], settings.title_words):
                continue
            keep, gone = min(ra, rb), max(ra, rb)
            parent[gone] = keep
            members[keep] += members.pop(gone)
    groups: dict[str, list[_Occurrence]] = {}
    for occurrence in occurrences:
        groups.setdefault(root(occurrence.name_key), []).append(occurrence)
    return [_Cluster(tuple(group)) for _, group in sorted(groups.items())]


def _judge_id(cluster: _Cluster) -> str:
    first = cluster.occurrences[0]
    apart = any(o.apart for o in cluster.occurrences)  # a judge a person kept apart, even after names join it
    return stable_id("judge", first.synthetic, key_text(first.court), sorted(cluster.name_keys), apart)


def _candidate(occurrence: _Occurrence, judge: _Cluster | None, score: float, reason: str, names: str | None = None) -> AliasCandidate:
    """``names`` lists the possible judges when the name fits several (no single candidate)."""
    return AliasCandidate(
        case_id=occurrence.case_id,
        raw_name=occurrence.raw_name,
        court=occurrence.court,
        synthetic=occurrence.synthetic,
        candidate_judge_id=None if judge is None else _judge_id(judge),
        candidate_name=names if judge is None else judge.display_name,
        score=round(score, 1),
        reason=reason,
    )


def _periods(cluster: _Cluster, tolerance_days: int) -> tuple[_Cluster, list[AliasCandidate]]:
    """Cases years apart from the judge's other cases at the same court may be another judge of the
    same name: the period with the most cases stays, the others wait for a decision."""
    dated = sorted((o for o in cluster.occurrences if o.dates), key=lambda o: (o.dates[0], o.case_id))
    if not dated:
        return cluster, []
    periods, last = [[dated[0]]], dated[0].dates[-1]
    for occurrence in dated[1:]:
        if (occurrence.dates[0] - last).days > tolerance_days:
            periods.append([])
        periods[-1].append(occurrence)
        last = max(last, occurrence.dates[-1])
    if len(periods) == 1:
        return cluster, []
    main = max(range(len(periods)), key=lambda i: (len({o.case_id for o in periods[i]}), -i))
    others = [o for i, period in enumerate(periods) if i != main for o in period]
    kept = _Cluster(tuple(o for o in cluster.occurrences if o not in others))
    years = round(tolerance_days / _DAYS_PER_YEAR)
    reason = f"same name at the same court, more than {years} years apart from the judge's other cases"
    return kept, [_candidate(o, kept, 100.0, reason) for o in others]


def _resemblance(cluster: _Cluster, judges: Sequence[_Cluster], settings: JudgeSettings) -> tuple[_Cluster, float, str] | None:
    """The registered judge this name may belong to, with the reason it waits; None when it resembles none."""
    near = []
    for judge in judges:
        if _conflicting(cluster.name_keys, judge.name_keys, settings.title_words):
            continue  # different initials: a different person
        score = _score(cluster.name_keys, judge.name_keys)
        if score >= settings.ambiguous_floor:
            reason = f"name close to {judge.display_name} at the same court (similarity {score:.0f}, below {settings.name_match_threshold:g})"
            near.append((score, judge, reason))
        elif _contained(cluster.name_keys, judge.name_keys, settings.title_words):
            near.append((score, judge, f"the name may be a shorter or longer form of {judge.display_name}'s name"))
    if not near:
        return None
    score, judge, reason = max(near, key=lambda item: (item[0], item[1].cases))
    return judge, score, reason


def _full_names(occurrences: Sequence[_Occurrence], settings: JudgeSettings) -> tuple[list[_Cluster], list[AliasCandidate]]:
    registered: list[_Cluster] = []
    pending: list[AliasCandidate] = []
    for cluster in sorted(_clusters(occurrences, settings), key=lambda c: (-c.cases, sorted(c.name_keys))):
        found = _resemblance(cluster, registered, settings)
        if found is not None:
            judge, score, reason = found
            pending += [_candidate(o, judge, score, reason) for o in cluster.occurrences]
            continue
        kept, waiting = _periods(cluster, settings.date_tolerance_days)
        registered.append(kept)
        pending += waiting
    return registered, pending


def _initials(occurrences: Sequence[_Occurrence], registered: list[_Cluster], settings: JudgeSettings) -> tuple[list[_Cluster], list[AliasCandidate]]:
    judges = list(registered)
    pending: list[AliasCandidate] = []
    titles = settings.title_words
    for group in sorted(_clusters(occurrences, settings), key=lambda c: (-c.cases, sorted(c.name_keys))):
        cores = {_core(key, titles) for key in group.name_keys}
        fits = [
            judge for judge in judges
            if any(_initials_fit(core, {_core(k, titles) for k in judge.name_keys}, settings.ambiguous_floor) for core in cores)
        ]  # fmt: skip
        if not fits:
            found = _resemblance(group, judges, settings)
            if found is None:
                judges.append(group)
            else:
                judge, score, reason = found
                pending += [_candidate(o, judge, score, reason) for o in group.occurrences]
            continue
        judge = fits[0] if len(fits) == 1 else None
        reason = "written with initials only" + ("" if judge else f"; it fits {len(fits)} judges at this court")
        score = _score(group.name_keys, judge.name_keys) if judge else 0.0
        names = None if judge else " or ".join(sorted(fit.display_name for fit in fits))
        pending += [_candidate(o, judge, score, reason, names) for o in group.occurrences]
    return judges[len(registered) :], pending


def _near_in_time(occurrence: _Occurrence, judge: _Cluster, tolerance_days: int) -> bool:
    dates = [d for o in judge.occurrences for d in o.dates]
    return any(abs((mine - theirs).days) <= tolerance_days for mine in occurrence.dates for theirs in dates)


def _attach_decided(decided: Sequence[_Occurrence], judges: list[_Cluster], tolerance_days: int) -> tuple[list[_Cluster], list[AliasCandidate], list[DataNote]]:
    """A person's 'same as' joins the one judge registered at the court under that name (two judges of
    the same name are told apart by the period of their cases); otherwise the name keeps waiting and a
    note says why. A decision never creates a judge of its own."""
    extra: dict[int, list[_Occurrence]] = {}
    pending, notes = [], []
    for occurrence in decided:
        matches = [index for index, judge in enumerate(judges) if occurrence.name_key and occurrence.name_key in judge.name_keys]
        if len(matches) > 1:
            matches = [index for index in matches if _near_in_time(occurrence, judges[index], tolerance_days)] or matches
        if len(matches) == 1:
            extra.setdefault(matches[0], []).append(occurrence)
            continue
        which = "names only a title" if not occurrence.name_key else "matches no judge" if not matches else f"matches {len(matches)} judges"
        pending.append(_candidate(occurrence, None, 0.0, f"the recorded decision {which} registered at this court"))
        text = (
            f"alias_decisions.yaml: in case {occurrence.case_id}, {occurrence.raw_name!r} is decided to be "
            f"{occurrence.decided_as!r}, which {which} registered at {occurrence.court}, so the name still waits"
        )
        notes.append(DataNote(text=text, synthetic=occurrence.synthetic))
    merged = [_Cluster(judge.occurrences + tuple(extra.get(index, ()))) for index, judge in enumerate(judges)]
    return merged, pending, notes


def _register_place(occurrences: Sequence[_Occurrence], settings: JudgeSettings) -> tuple[list[_Cluster], list[AliasCandidate], list[DataNote]]:
    """Judges and pending names at one court, for one kind of data."""
    apart_keys = sorted({o.name_key for o in occurrences if o.apart and o.name_key})
    apart = [_Cluster(tuple(o for o in occurrences if o.apart and o.name_key == key)) for key in apart_keys]
    undecided = [o for o in occurrences if not o.apart and o.decided_as is None]
    unnamed = [o for o in undecided if not o.name_key]
    pending = [_candidate(o, None, 0.0, "the name is only a title, with no personal name to match") for o in unnamed]
    named = [o for o in undecided if o.name_key]
    titles = settings.title_words
    registered, waiting = _full_names([o for o in named if not _initials_only(o.name_key, titles)], settings)
    initial_judges, waiting_initials = _initials([o for o in named if _initials_only(o.name_key, titles)], registered, settings)
    decided = [o for o in occurrences if o.decided_as is not None]
    judges, waiting_decided, notes = _attach_decided(decided, registered + initial_judges + apart, settings.date_tolerance_days)
    return judges, pending + waiting + waiting_initials + waiting_decided, notes


def _judge(cluster: _Cluster) -> RegisteredJudge:
    first = min(cluster.occurrences, key=lambda o: o.case_id)
    return RegisteredJudge(
        judge_id=_judge_id(cluster),
        display_name=cluster.display_name,
        court=first.court,
        synthetic=first.synthetic,
        variants=tuple(sorted({o.raw_name for o in cluster.occurrences})),
        case_ids=tuple(sorted({o.case_id for o in cluster.occurrences})),
    )


def build_registry(records: Sequence[CaseRecord], decisions: Sequence[AliasDecision], settings: JudgeSettings) -> Registry:
    occurrences, notes = _occurrences(records, decisions, settings.title_words)
    clusters: list[_Cluster] = []
    pending: list[AliasCandidate] = []
    for place in sorted({o.place for o in occurrences}):
        registered, waiting, place_notes = _register_place([o for o in occurrences if o.place == place], settings)
        clusters += registered
        pending += waiting
        notes += place_notes
    judges = sorted((_judge(cluster) for cluster in clusters), key=lambda j: (j.display_name, j.court, j.judge_id))
    assignment = {(o.case_id, o.raw_name): _judge_id(cluster) for cluster in clusters for o in cluster.occurrences}
    return Registry(
        judges=tuple(judges),
        pending=tuple(sorted(pending, key=lambda c: (c.case_id, c.raw_name))),
        assignment=MappingProxyType(assignment),
        notes=tuple(notes),
    )
