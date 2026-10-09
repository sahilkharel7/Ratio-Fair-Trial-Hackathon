import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  allFindings,
  currentReviews,
  reviewError,
  makeReview,
  reviewWorksheet,
} from "../src/workflow.mjs";
const data = JSON.parse(
  readFileSync(new URL("../public/demo.json", import.meta.url), "utf8"),
);
const flag = allFindings(data)[0];
const documents = Object.fromEntries(
  [data.record, ...data.history]
    .flatMap((r) => r.documents)
    .map((d) => [d.id, d]),
);

test("review decisions require reasons and revised wording, while acceptance may be unqualified", () => {
  assert.equal(reviewError(flag, "accepted", "", ""), "");
  assert.match(reviewError(flag, "rejected", "", ""), /reason/);
  assert.match(
    reviewError(flag, "edited", "Reason", flag.message),
    /unchanged/,
  );
  assert.match(
    reviewError(
      flag,
      "edited",
      "Reason",
      "The judge is corrupt.",
      data.messages.block_list,
    ),
    /character/,
  );
  assert.match(
    reviewError(
      flag,
      "edited",
      "Reason",
      "The judge is c\u200borrupt.",
      data.messages.block_list,
    ),
    /character/,
  );
  assert.match(
    reviewError(
      flag,
      "edited",
      "Reason",
      "The judge is cоrrupt.",
      data.messages.block_list,
      data.review_transliteration,
    ),
    /character/,
  ); // Cyrillic о
  assert.equal(
    reviewError(
      flag,
      "edited",
      "Source-based revision",
      "The notes record the refusal of an adjournment.",
      data.messages.block_list,
    ),
    "",
  );
});
test("reopening removes the current decision without losing its history or source snapshot", () => {
  const first = makeReview(
    data,
    flag,
    "accepted",
    "Demo check",
    "",
    "",
    "2026-10-09T00:00:00Z",
  );
  const edited = makeReview(
    data,
    flag,
    "edited",
    "Use neutral wording",
    "The notes record a refusal.",
    "",
    "2026-10-09T01:00:00Z",
  );
  const reopened = makeReview(
    data,
    flag,
    "reopened",
    "",
    "",
    "",
    "2026-10-09T02:00:00Z",
  );
  const history = [first, edited, reopened];
  assert.equal(currentReviews(history)[flag.id], undefined);
  assert.equal(history.length, 3);
  assert.deepEqual(edited.flag.evidence, flag.evidence);
  assert.equal(first.edited_message, null);
});
test("the browser worksheet preserves original findings, revised wording, source references and withdrawn issues", () => {
  const review = makeReview(
    data,
    flag,
    "edited",
    "Demo check",
    "The notes record a refusal.",
    "",
    "2026-10-09T01:00:00Z",
  );
  const worksheet = reviewWorksheet(
    data,
    [review],
    [
      { module: "absence", standard: "Article 14", note: "Visible issue" },
      {
        module: "absence",
        standard: "Article 9",
        note: "Withdrawn issue",
        withdrawn_at: "2026-10-09",
      },
    ],
    documents,
  );
  assert.ok(worksheet.includes(flag.message));
  assert.ok(
    worksheet.includes("Reviewer’s wording: The notes record a refusal."),
  );
  assert.ok(worksheet.includes("line 13"));
  assert.ok(worksheet.includes("Visible issue"));
  assert.ok(!worksheet.includes("Withdrawn issue"));
  assert.ok(worksheet.includes("not synchronized"));
  assert.ok(worksheet.includes("Decision history"));
});
