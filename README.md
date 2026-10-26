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

## Status (first version)

Tested end-to-end against one real respondent's scan (`BP_15.pdf`) from the
pilot batch:

- OCR reliably reads the printed code when the row crop is correctly
  aligned (validated directly against Tesseract: `ER_1`–`ER_5` all read
  correctly with 3x upscaling + binarization + a restricted character
  whitelist).
- Row-boundary detection (finding where each table row starts/ends) and
  column-boundary detection (finding the Code column's left/right edges)
  are the less reliable parts — both needed several rounds of calibration
  against this real scan (see `ocr_pipeline.py` docstrings for what was
  tried and why) and are still imperfect: some rows are missed entirely,
  and OCR doesn't confidently read every row's code even when the row
  itself is correctly found.
- Net result on BP_15.pdf: 59 of 71 items got a value (vs. 68 of 71 from
  the original pipeline's blind count-and-flatten approach), but with
  correct localization — e.g. the original pipeline always blamed the
  *last* 3 items (PC_3/4/5) for any shortfall; this version correctly
  identifies PC_5 as read and PC_3/4 as the genuine gap.

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
