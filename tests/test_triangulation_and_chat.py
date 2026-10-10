"""Triangulation and the case chat, on the synthetic Sorin matter and a planted synthetic reference corpus.

Retrieval uses a deterministic bag-of-words embedder and a fake corpus; the model is a fake that obeys
the schema, except in the network test, which drives the real Ollama client through a fake transport.
"""

import json
import re
import zlib

import numpy as np
import pytest

from ratio.case_chat import BANNER, LABEL, REFUSAL, ask
from ratio.config import load_config
from ratio.library import LibraryStore
from ratio.llm import OllamaClient, OllamaUnavailable
from ratio.store import CaseStore
from ratio.triangulation import POINTER_LABEL, rights, triangulate
from ratio.workspace import seed_collection

# SYNTHETIC reference passages: invented decisions, no real case.
REFERENCES = {
    "ref-burden": ("Planted v. Burden (synthetic)", "The Committee recalls that the burden of proof of the charge is on the prosecution. The court required the author to prove that she had not organised the assembly, which reversed the presumption of innocence."),
    "ref-interpreter": ("Planted v. Interpreter (synthetic)", "The author did not understand the language of the court, and no interpreter was provided at any hearing."),
    "ref-delay": ("Planted v. Delay (synthetic)", "The author was held in police custody for seven days before he was brought before a judge."),
    "ref-private": ("Private collection document (synthetic)", "The accused must prove that she did not organise the assembly; the burden of proof was reversed."),
}


def embed(texts):
    rows = []
    for text in texts:
        vector = np.zeros(512)
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[zlib.crc32(word.encode()) % 512] += 1
        norm = np.linalg.norm(vector)
        rows.append(vector / norm if norm else vector)
    return np.array(rows)


class FakeResearch:
    def embed(self, texts):
        return embed(texts)

    def search(self, query, *, mode="phrase", limit=20, kind="", state=""):
        assert mode == "meaning" and len(query) <= 300
        ids = list(REFERENCES)
        scores = embed([query] + [REFERENCES[i][1] for i in ids])
        order = np.argsort(-(scores[1:] @ scores[0]), kind="stable")
        hits = [
            {
                "document": {"id": ids[i], "title": REFERENCES[ids[i]][0], "kind": "ccpr_views", "private": ids[i] == "ref-private"},
                "start": 0,
                "end": len(REFERENCES[ids[i]][1]),
                "text": REFERENCES[ids[i]][1],
            }
            for i in order
        ]
        return {"hits": hits[:limit]}


class SchemaLLM:
    """Answers with the first allowed ids; records every request."""

    def __init__(self, chat=None):
        self.calls, self.chat = [], chat

    def complete_json(self, *, system, user, schema, purpose):
        payload = json.loads(user)
        self.calls.append((purpose, payload))
        if purpose == "triangulation":
            planted = next(p for p in payload["past_case_passages"] if p["case"].startswith("Planted v. Burden"))
            facts = [{
                "case_passage": payload["open_case_passages"][0]["id"],
                "reference_passage": planted["id"],
                "fact": "The accused had to prove she did not organise the assembly.",
                "why_relevant": "Both put the burden of proof on the defence.",
                "comparison": "similar",
                "explanation": "In both, the court required the accused to disprove the charge.",
            }]
            return schema.model_validate({"facts": facts})
        return schema.model_validate(self.chat(payload))


@pytest.fixture
def library(tmp_path):
    store = LibraryStore(CaseStore(tmp_path / "cases.db"))
    seed_collection(store)
    return store


def text_of(library, span):
    record = library.cases.load_case("sorin-2025")
    return next(d.text for d in record.documents if d.id == span["doc_id"])[span["start"]: span["end"]]


def test_triangulation_finds_the_planted_case_and_every_fact_has_source_spans(library):
    llm = SchemaLLM()
    result = triangulate(library, FakeResearch(), "sorin-2025", "14(2)", llm=llm)

    top = result["similar_cases"][0]
    assert top["reference"]["id"] == "ref-burden"
    assert "Burden of proof put on the defence" in top["shared_indicators"] and top["cites_right"]
    assert "ref-private" not in {c["reference"]["id"] for c in result["similar_cases"]}
    assert result["label"] == POINTER_LABEL and result["label"].startswith("Similarity is a pointer, not a finding")
    assert all(isinstance(c["score"], float) for c in result["similar_cases"])
    facts = [f for c in result["similar_cases"] for f in c["facts"]]
    assert facts
    for fact in facts:
        assert fact["case_span"]["text"] == text_of(library, fact["case_span"])
        reference = fact["reference_span"]
        planted = REFERENCES[reference["doc_id"].removeprefix("precedent:")][1]
        assert reference["text"] == planted[reference["start"]: reference["end"]]
    assert [purpose for purpose, _ in llm.calls] == ["triangulation"]


def test_triangulation_without_the_local_model_still_ranks_and_says_so(library):
    class Down:
        def complete_json(self, **kwargs):
            raise OllamaUnavailable("Ollama is not running")

    result = triangulate(library, FakeResearch(), "sorin-2025", "14(2)", llm=Down())
    assert result["similar_cases"][0]["reference"]["id"] == "ref-burden"
    assert all(c["facts"] == [] for c in result["similar_cases"])
    assert "unavailable" in result["note"]


def test_every_listed_right_can_be_triangulated_and_unknown_rights_are_refused(library):
    for right in rights():
        result = triangulate(library, FakeResearch(), "sorin-2025", right["id"], llm=SchemaLLM())
        assert result["right"]["id"] == right["id"]
    with pytest.raises(ValueError):
        triangulate(library, FakeResearch(), "sorin-2025", "99(9)", llm=SchemaLLM())
    with pytest.raises(KeyError):
        triangulate(library, FakeResearch(), "no-such-case", "14(2)", llm=SchemaLLM())


def test_chat_answer_cites_exact_sources(library):
    llm = SchemaLLM(chat=lambda p: {"supported": True, "answer": "The accused had to prove she did not organise the assembly.", "citations": [p["passages"][0]["id"]]})
    reply = ask(library, FakeResearch(), "sorin-2025", "Who had to prove what?", llm=llm)
    assert reply["supported"] and reply["label"] == LABEL and reply["banner"] == BANNER
    assert reply["citations"]
    for citation in reply["citations"]:
        span = citation["span"]
        if citation["kind"] == "case":
            assert span["text"] == text_of(library, span)
        else:
            planted = REFERENCES[span["doc_id"].removeprefix("precedent:")][1]
            assert span["text"] == planted[span["start"]: span["end"]]
    sent = llm.calls[0][1]["passages"]
    assert sent[0]["id"].startswith("c")  # the open case's own passages come first
    assert all("Private collection" not in p["source"] for p in sent)


def test_chat_refuses_unsupported_and_uncited_answers(library):
    unsupported = SchemaLLM(chat=lambda p: {"supported": False, "answer": "", "citations": []})
    uncited = SchemaLLM(chat=lambda p: {"supported": True, "answer": "Probably yes.", "citations": []})
    for llm in (unsupported, uncited):
        reply = ask(library, FakeResearch(), "sorin-2025", "What is the defendant's favourite food?", llm=llm)
        assert reply == {"supported": False, "answer": REFUSAL, "citations": [], "label": LABEL, "banner": BANNER}
    with pytest.raises(ValueError):
        ask(library, FakeResearch(), "sorin-2025", " ", llm=unsupported)
    with pytest.raises(ValueError):
        ask(library, FakeResearch(), "sorin-2025", "x" * 501, llm=unsupported)


BLOCKED = "The biased judge lied and was corrupt."


class RudeLLM(SchemaLLM):
    """Replies in words that characterise a person (hard rule 3 must remove them)."""

    def complete_json(self, *, system, user, schema, purpose):
        payload = json.loads(user)
        if purpose == "triangulation":
            planted = next(p for p in payload["past_case_passages"] if p["case"].startswith("Planted v. Burden"))
            fact = {"case_passage": payload["open_case_passages"][0]["id"], "reference_passage": planted["id"],
                    "fact": BLOCKED, "why_relevant": BLOCKED, "comparison": "similar", "explanation": BLOCKED}
            return schema.model_validate({"facts": [fact]})
        return schema.model_validate({"supported": True, "answer": BLOCKED, "citations": [payload["passages"][0]["id"]]})


def test_model_text_never_shows_words_that_characterise_a_person(library):
    fact = triangulate(library, FakeResearch(), "sorin-2025", "14(2)", llm=RudeLLM())["similar_cases"][0]["facts"][0]
    reply = ask(library, FakeResearch(), "sorin-2025", "Who had to prove what?", llm=RudeLLM())
    for text in (fact["fact"], fact["why_relevant"], fact["explanation"], reply["answer"]):
        assert "biased" not in text and "corrupt" not in text


def test_a_reply_that_fails_the_schema_is_a_refusal_not_an_outage(library):
    from ratio.llm import LLMResponseError

    class Garbled:
        def complete_json(self, **kwargs):
            raise LLMResponseError("citations ['c99'] are not allowed")

    reply = ask(library, FakeResearch(), "sorin-2025", "Who had to prove what?", llm=Garbled())
    assert reply["answer"] == REFUSAL and not reply["citations"]


def test_a_very_long_sentence_is_cut_into_bounded_passages():
    from types import SimpleNamespace

    from ratio.triangulation import PASSAGE_CHARS, case_passages

    text = "\n".join(f"line {i} of an OCR page without full stops" for i in range(400))
    record = SimpleNamespace(documents=[SimpleNamespace(id="scan.pdf", title="Scan", text=text)])
    passages = case_passages(record)
    assert passages and all(len(p["text"]) <= PASSAGE_CHARS for p in passages)
    assert all(p["text"] == text[p["start"]: p["end"]] for p in passages)


def test_chat_talks_only_to_the_local_ollama_server(library):
    requests = []

    def transport(url, body, timeout):
        requests.append(url)
        if url.endswith("/api/show"):
            return {"details": {}}
        content = json.dumps({"supported": True, "answer": "She had to prove she did not organise it.", "citations": ["c1"]})
        return {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop"}

    client = OllamaClient(load_config().settings.llm, model="qwen2.5:7b-instruct", transport=transport)
    reply = ask(library, FakeResearch(), "sorin-2025", "Who had to prove what?", llm=client)
    assert reply["supported"]
    assert requests and all(url.startswith("http://127.0.0.1:11434/") for url in requests)
