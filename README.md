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

Tested end-to-end against one real respondent's scan (`BP_15.pdf`) from the
pilot batch, through several rounds of tracing actual failures against this
real scan (see `ocr_pipeline.py` docstrings for the specific evidence
behind each calibration choice — row/column boundary detection in
particular went through multiple wrong approaches before landing on ones
verified against this document):

- Row-boundary detection (Code-column text-blob clustering) recovers
  **71 of 71 real rows exactly** on this document — earlier approaches
  based on detecting the table's horizontal rule lines directly (tried two
  different ways) recovered 62-80 depending on the half, both too few and
  too many.
- Column-boundary detection (finding the Code column's left/right edges)
  is calibrated against 9 real halves of this document and is reliable in
  the same run.
- Alignment correctness was the main remaining issue: an earlier version
  silently mislabeled items when a row went undetected nearby (verified:
  PC_3/PC_4's real marks were being attributed to PC_1/PC_2's codes). This
  is fixed — mismatched stretches are now reported explicitly ("N rows
  found for M expected items between X and Y") instead of guessed. A
  related bug (a 3-character garbled OCR read fuzzy-matching a code 27
  positions away) is also fixed with a plausible-jump-distance cap.
- Net result on BP_15.pdf: **41 of 71 items resolved with a real value**,
  all verified correctly positioned (including PC_3 and PC_4, the case
  that originally exposed the mislabeling bug). The remaining 30 are
  stretches where OCR didn't produce enough anchors to safely interpolate
  — reported honestly as "check this stretch manually" rather than guessed.

**This has not yet been validated against the rest of the pilot batch**
(BP_9, 10, 11, 13, 14) the way the original pipeline was. Expect further
calibration to be needed, the same way the original pipeline needed
several rounds of real-scan testing.

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
