import { test } from "node:test";
import assert from "node:assert/strict";
import {
  matchesQuery,
  firstMatch,
  sourceKey,
  sourceCitation,
} from "../src/research.mjs";

test("multiword searches require every word and preserve quoted phrases", () => {
  assert.equal(
    matchesQuery("The defence requested time to prepare.", "defence prepare"),
    true,
  );
  assert.equal(
    matchesQuery(
      "The defence requested time to prepare.",
      "defence interpreter",
    ),
    false,
  );
  assert.equal(
    matchesQuery(
      "First appearance before the judge",
      '"first appearance" judge',
    ),
    true,
  );
  assert.equal(
    matchesQuery("First judicial appearance", '"first appearance"'),
    false,
  );
  assert.equal(matchesQuery("Any document", ""), true);
});

test("search results use original source text and Unicode character offsets", () => {
  const document = {
    id: "example",
    title: "Judgment",
    text: "İ ⚖️ 😀\nThe DEFENCE requested time.",
  };
  const span = firstMatch(document, '"defence" time');
  assert.equal(span.text, "DEFENCE");
  assert.equal(
    Array.from(document.text).slice(span.start, span.end).join(""),
    span.text,
  );
  assert.equal(
    sourceCitation(document, span),
    "Judgment, line 2 [synthetic record]",
  );
  assert.notEqual(sourceKey(document), sourceKey(document, span));
  assert.equal(firstMatch(document, "missing"), null);
});
