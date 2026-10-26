# Questionnaire Digitizer — OCR-anchored

A separate, from-scratch implementation of the checkbox-digitizing pipeline
used in `questionnaire-digitizer`, built to solve a specific limitation of
that tool: when the detected mark count doesn't match the schema's expected
count, the original pipeline has no way to know *where* in the document the
gap occurred, so every item past the shortfall gets blamed as "not
detected," even when the true gap was earlier in the document and those
items were actually read correctly.

This version reads the printed item code next to each row (e.g. `"PC_3"`)
directly off the scan via Tesseract OCR, and matches it against the
schema's known, finite code list. That ties every row — whether or not a
mark was found on it — to its real schema position *before* any sequential
mark-counting happens, which is what actually fixes the localization
problem, not just the symptom.

## Status

Validated end-to-end against all 6 real respondent scans in the pilot
batch (BP_9, 10, 11, 13, 14, 15), not just one — several calibration
choices made against BP_15 alone turned out not to generalize and were
caught and fixed by this broader validation (see `ocr_pipeline.py`
docstrings for the specific evidence behind each one: a fixed Code-column
darkness threshold, a fixed row-density cutoff, and a fixed OCR
binarization threshold all failed on at least one other document before
being replaced with per-document-adaptive versions).

Items resolved with a real, verified-correct value (of 71):

| Document | Resolved |
|---|---|
| BP_15 | 51 |
| BP_9  | 46 |
| BP_10 | 44 |
| BP_14 | 33 |
| BP_13 | 31 |
| BP_11 | 27 |

Key fixes along the way:
- Row-boundary detection (Code-column text-blob clustering, not the
  table's rule lines - those proved too thin/broken to detect reliably in
  two different attempts) recovers close to 71 of 71 rows on most
  documents, using a per-half adaptive density cutoff (60th percentile of
  the row's own density profile) rather than one fixed threshold - a
  fixed threshold or a median-based one both failed differently on real
  documents (BP_13's background noise, BP_11's heavy bleed-through).
- Column-boundary detection (the Code column's left/right edges) is
  template-based - proportions of the answer grid's own position - and
  generalizes well since every respondent fills the same printed form.
- Answer-grid line detection retries at progressively lower thresholds
  when the original pipeline's calibration finds nothing - recovered 8 of
  9 grids on BP_11.pdf, whose lines are noticeably fainter than the rest
  of the batch.
- OCR's binarization step adapts per-crop (Otsu threshold) instead of a
  fixed gray cutoff, after one found blank (zero ink crossed the old fixed
  threshold) on a real document.
- The Sr.No column (plain digits, e.g. "69") is read as a second,
  independent position anchor alongside the Code column - digits read far
  more reliably than the Code column's mixed letters/underscore/digits,
  which need an exact or near-exact fuzzy match to be usable at all. This
  was the single largest improvement of the project: BP_11 alone went
  from 2 to 27 resolved items.
- Alignment no longer silently mislabels items when a row goes
  undetected nearby (the bug that originally exposed this whole
  investigation - PC_3/PC_4's real marks were being attributed to
  PC_1/PC_2's codes). Mismatched stretches are now reported explicitly
  ("N rows found for M expected items between X and Y") instead of
  guessed.

All six documents are now in a broadly similar, meaningfully useful range
(27-51 of 71). None is "done" - the remaining gaps on every document are
honestly flagged for manual review rather than guessed, consistent with
this project's whole design goal.

## Architecture

- `ocr_pipeline.py` — grid/row/column detection, OCR, and OCR-anchored
  alignment (the core of this rewrite; see its module and function
  docstrings for the detailed reasoning behind each calibration choice).
- `db.py` — SQLite storage (schema auto-synced from `schemas/*.json` on
  startup; responses are expected to live for one working session, same
  as the original app).
- `app.py` — Streamlit review UI.

## Deployment

Same as the original app: push to `main`, Streamlit Community Cloud
auto-redeploys. `packages.txt` adds `tesseract-ocr` (OCR) and
`poppler-utils` (PDF rendering) as apt dependencies.

Tesseract (not a cloud OCR API) was chosen deliberately: this data is from
minors in a research study, so keeping OCR local to the deployment rather
than sending scanned response sheets to a third-party service matters.
