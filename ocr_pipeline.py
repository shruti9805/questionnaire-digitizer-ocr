"""Checkbox-grid detection with OCR-anchored row identification.

Builds on the same proven geometry as questionnaire-digitizer's
checkbox_pipeline.py (vertical gridline detection via longest-contiguous-
dark-run projection, HSV-saturation ink isolation with per-document Otsu
calibration) but adds a second axis of detection: horizontal row
boundaries, and OCR of the printed item "Code" column next to each row.

Why: the original pipeline has no way to know which schema item a detected
mark belongs to except by counting marks in document order and zipping
them against the schema list. When the detected count falls short of the
expected count (common with faint pen strokes), that design has no way to
localize WHERE the gap is - every item past the shortfall gets padded as
"not detected" even if the true gap was earlier in the document. Reading
the printed code (e.g. "PC_3") directly off each row removes that
ambiguity: every row - whether or not a mark was found on it - gets tied
to its real schema position before any mark counting happens.

Design: OCR is not trusted as sole ground truth (scanned paper forms OCR
imperfectly). Each row's OCR'd code is fuzzy-matched against the schema's
known, finite code list and only accepted if it's a close match AND its
schema position is not behind the current read-through cursor. A row whose
OCR reading fails or produces a weak match falls back to "the position
right after the last confirmed anchor" - the same sequential assumption
the original design relied on everywhere, but now re-synced at almost
every row instead of never, which is what actually fixes the localization
problem.
"""
from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

# ---------- PDF rendering ----------

def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "page"
    result = subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), pdf_path, str(prefix)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr}")
    pages = sorted(out.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
    return [str(p) for p in pages]


def split_halves(page_png_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Scanned booklet pages are rendered as one wide image per physical
    sheet (two pages side by side); (1,1,3) sentinel means "no left half"
    for a page that wasn't a two-up scan."""
    im = Image.open(page_png_path).convert("RGB")
    w, h = im.size
    if w < h:
        return np.zeros((1, 1, 3), dtype=np.uint8), np.array(im)
    left = np.array(im.crop((0, 0, w // 2, h)))
    right = np.array(im.crop((w // 2, 0, w, h)))
    return left, right


# ---------- line detection (shared by both axes: transpose for rows) ----------

def _longest_run(mask_1d: np.ndarray) -> tuple[int, int, int]:
    best_len = cur_len = 0
    best_start = cur_start = 0
    for i, v in enumerate(mask_1d):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_len, best_start, best_start + best_len


def _find_lines(dark: np.ndarray, axis_len: int, scan_len: int, min_run_frac: float,
                 scan_range: tuple[int, int]) -> list[tuple[int, int, int]]:
    """dark: 2D boolean array. axis=0 scans columns (vertical lines), the
    caller transposes `dark` to scan rows (horizontal lines) instead.
    Returns (position, extent_start, extent_end) tuples, position in the
    scanned axis, extent in the other axis."""
    min_run = int(axis_len * min_run_frac)
    runs = {}
    for pos in range(*scan_range):
        length, start, end = _longest_run(dark[:, pos])
        if length > min_run:
            runs[pos] = (length, start, end)
    if not runs:
        return []
    xs = sorted(runs.keys())
    lines = []
    cur_xs, cur_starts, cur_ends = [xs[0]], [runs[xs[0]][1]], [runs[xs[0]][2]]
    for x in xs[1:]:
        if x - cur_xs[-1] <= 3:
            cur_xs.append(x)
            cur_starts.append(runs[x][1])
            cur_ends.append(runs[x][2])
        else:
            lines.append((int(np.mean(cur_xs)), int(np.mean(cur_starts)), int(np.mean(cur_ends))))
            cur_xs, cur_starts, cur_ends = [x], [runs[x][1]], [runs[x][2]]
    lines.append((int(np.mean(cur_xs)), int(np.mean(cur_starts)), int(np.mean(cur_ends))))
    return lines


@dataclass
class GridLayout:
    answer_left: int
    answer_right: int
    col_bounds: list[float]
    code_x0: int  # left edge of the Code column
    code_x1: int  # right edge of the Code column (== answer table's left margin area end)
    y_top: int
    y_bottom: int


def find_grid_layout(rgb: np.ndarray, min_run_frac: float = 0.08) -> GridLayout | None:
    """Finds the 6 vertical lines bounding the 5 answer columns (searching
    only the right half of the half-page image, where the answer grid
    lives - verified against real scans, same as the original pipeline),
    then separately finds the Sr.No|Code and Code|Statement boundary lines
    by searching the full width for every vertical line and picking the
    two immediately left of the answer grid's left edge.

    The original min_run_frac=0.08 contiguous-dark-run requirement fails
    outright (zero lines found) on a real scan with noticeably fainter
    grid lines than the rest of the pilot batch (BP_11.pdf) - its longest
    contiguous run anywhere in the grid's true column range was only ~7%
    of the half's height, just under the 8% floor. Retried at
    progressively lower fractions (same contiguous-run mechanism, not a
    different detection method) only when the stricter one finds no
    width-valid grid - recovers a valid grid on 8 of 9 of that document's
    halves. The already-working documents are unaffected since 0.08 is
    always tried first and used whenever it already succeeds."""
    h, w, _ = rgb.shape
    gray = np.array(Image.fromarray(rgb).convert("L"))
    dark = gray < 235

    x0 = int(w * 0.5)
    xs_only: list[int] = []
    for frac in (min_run_frac, 0.05, 0.03, 0.02):
        grid_lines = _find_lines(dark, h, w, frac, (x0, w))
        if len(grid_lines) < 2:
            continue
        candidate_xs = sorted(l[0] for l in grid_lines)
        grid_width = candidate_xs[-1] - candidate_xs[0]
        if 0.12 * w <= grid_width <= 0.26 * w:
            xs_only = candidate_xs
            break
    if not xs_only:
        return None
    left, right = xs_only[0], xs_only[-1]
    col_bounds = xs_only if len(xs_only) == 6 else [left + (right - left) * i / 5 for i in range(6)]

    # Vertical lines left of the grid (Sr.No|Code and Code|Statement
    # boundaries), restricted to the table's own y-span. A single global
    # density cutoff across the whole pre-grid width proved unreliable
    # (verified on real halves of BP_15.pdf): the true lines are faint
    # enough that a cutoff loose enough to catch them also catches
    # Hindi-text density further right (the statement column's headstroke,
    # see find_row_bands), while a cutoff tight enough to exclude that
    # noise excludes some genuine lines too.
    #
    # What worked instead: each boundary line's x-position, as a fraction
    # of `left` (the answer grid's own, reliably-found left edge), is
    # consistent across real halves - checked directly against 9 halves of
    # BP_15.pdf: the Sr.No|Code line sits at 0.17-0.23x`left`, the
    # Code|Statement line at 0.25-0.32x`left` (there is also a third,
    # fainter line further left around 0.09-0.14x`left` - the table's own
    # outer left border, not a column boundary - which an earlier version
    # of this search mistakenly paired with the Sr.No|Code line, cropping
    # the Sr.No column's digits instead of the Code column's text).
    # Searching each narrow proportional window independently for its
    # single highest-density point, rather than one global threshold
    # across the whole width, avoids Hindi-text density peaks entirely,
    # since those don't fall inside either window. A resulting width
    # outside the observed 82-92px range (two of 9 halves) means a window
    # didn't find its genuine line - the proportional midpoint estimate is
    # used instead rather than trusting a bad peak.
    MARGIN = 150
    y_top = max(0, min(l[1] for l in grid_lines) - MARGIN)
    y_bottom = max(l[2] for l in grid_lines) + MARGIN
    dark_table_rows = dark[y_top:y_bottom, :]
    col_dark_frac = dark_table_rows.mean(axis=0)

    def best_peak_x(frac_lo: float, frac_hi: float) -> int | None:
        x0, x1 = int(frac_lo * left), int(frac_hi * left)
        window = col_dark_frac[x0:x1]
        if len(window) == 0:
            return None
        return x0 + int(np.argmax(window))

    cand_x0 = best_peak_x(0.17, 0.23)
    cand_x1 = best_peak_x(0.25, 0.32)
    width = (cand_x1 - cand_x0) if (cand_x0 is not None and cand_x1 is not None) else None
    if width is not None and 65 <= width <= 105:
        code_x0, code_x1 = cand_x0, cand_x1
    else:
        code_x0, code_x1 = int(0.195 * left), int(0.285 * left)

    return GridLayout(answer_left=left, answer_right=right, col_bounds=col_bounds,
                       code_x0=code_x0, code_x1=code_x1, y_top=y_top, y_bottom=y_bottom)


def find_row_bands(rgb: np.ndarray, layout: GridLayout) -> list[tuple[int, int]]:
    """Row boundaries, found from the Code column's own printed text
    rather than the table's horizontal rule lines.

    Two earlier approaches at finding the rule lines directly both proved
    unreliable on real scans: a contiguous-dark-run test (the lines are
    thin and broken by scan noise - the longest unbroken run at a true
    rule-line y was only ~27% of the table width) and a density-peak test
    (text density and rule-line density ranges overlap too much on some
    halves to separate with one threshold). Both approaches' failures
    compounded into alignment errors downstream.

    What works instead: each row has exactly one short, isolated code
    label (e.g. "PC_3") printed near its top, surrounded by a large blank
    gap down to the next row's label - the statement text's own multi-line
    height is what makes that gap large, not anything about the rule
    lines. The border lines bounding the Code column itself must be
    excluded from this scan (an 8px inward margin) - included, a column
    border's own continuous height bridges every row's gap into one giant
    cluster.

    A single fixed darkness threshold for "is this pixel ink" does not
    generalize across different respondents' scans: one real scan
    (BP_13.pdf) had a low but non-zero background density of ~1-4%
    *everywhere* in the Code column (paper texture/scan noise, not text),
    which a threshold tuned loose enough to catch fainter ink elsewhere
    also picked up as one unbroken "row", while a threshold tight enough
    to reject it also rejected genuine but lighter ink on other documents.
    Nor is a fixed *multiple of the column's own median density* robust -
    tried first, it failed on a different real scan (BP_11.pdf) whose
    Code column carries unusually heavy bleed-through, pushing the median
    itself high enough that almost nothing cleared the resulting cutoff.

    What generalized, checked against row-detection totals across all 6
    real respondent scans gathered so far (71, 63-73, 40-60 depending on
    scan quality, vs. 71 expected on every document - tighter and more
    consistent than any fixed threshold tried): per row, compute the
    fraction of dark pixels across the Code column's width, then set the
    text-vs-background cutoff at the 60th percentile of that profile plus
    35% of the way to its peak. Using a percentile rather than the mean/
    median is what survives BP_11.pdf's heavy bleed-through - a row of
    genuine text is a small minority of any row's height, so even a noisy
    document's quieter majority of rows still anchors the low end of the
    profile, which the median (pulled toward dense content) could not."""
    h, w, _ = rgb.shape
    gray = np.array(Image.fromarray(rgb).convert("L"))
    x0, x1 = min(w, layout.code_x0 + 8), min(w, layout.code_x1 - 8)
    if x1 <= x0:
        return []
    dark = gray[layout.y_top:layout.y_bottom, x0:x1] < 235
    row_frac = dark.mean(axis=1)
    peak = row_frac.max()
    if peak < 0.03:
        return []
    baseline = np.percentile(row_frac, 60)
    cutoff = max(baseline + 0.35 * (peak - baseline), baseline + 0.03)
    is_text = row_frac > cutoff
    ys = np.where(is_text)[0]
    if len(ys) == 0:
        return []
    ys = ys + layout.y_top

    Y_GAP = 25
    blobs: list[tuple[int, int]] = []
    cur = [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= Y_GAP:
            cur.append(y)
        else:
            blobs.append((cur[0], cur[-1]))
            cur = [y]
    blobs.append((cur[0], cur[-1]))

    centers = [(a + b) // 2 for a, b in blobs]
    boundaries = [layout.y_top] + [
        (c1 + c2) // 2 for c1, c2 in zip(centers, centers[1:])
    ] + [layout.y_bottom]
    return list(zip(boundaries, boundaries[1:]))


# ---------- OCR ----------

CODE_PATTERN = re.compile(r"[A-Z]{1,3}_?\s?\d{1,2}")


def _preprocess_for_ocr(crop: Image.Image) -> Image.Image:
    """A fixed binarization threshold (previously gray>150) does not
    generalize: verified on a real scan (BP_13.pdf) whose code-text ink
    never got darker than 158, so every pixel fell on the "white" side of
    that fixed cutoff and the processed image was blank - OCR had nothing
    to read, independent of anything about row/column detection. Otsu's
    threshold, computed on this crop's own grayscale histogram, adapts
    to each respondent's actual ink darkness the same way ink-mask
    calibration already does elsewhere in this pipeline."""
    gray = crop.convert("L")
    scale = 3
    up = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    arr = np.array(up)
    normalized = arr.astype(float) / 255.0
    if normalized.max() - normalized.min() < 0.05:
        return Image.fromarray(np.full_like(arr, 255, dtype=np.uint8))
    otsu = _otsu_threshold(normalized, bins=50)
    bw = ((normalized > otsu) * 255).astype(np.uint8)
    return Image.fromarray(bw)


def ocr_row_code(rgb: np.ndarray, layout: GridLayout, y0: int, y1: int) -> str | None:
    h, w, _ = rgb.shape
    # +6 excludes the Sr.No|Code border line (confirmed bleeding into a
    # tight crop and confusing Tesseract's layout analysis); +18 beyond
    # the calibrated right edge gives trailing digits room - the
    # Code|Statement boundary was calibrated as a column-structure
    # estimate, not a pixel-exact text bound, and a real crop at exactly
    # that edge was found clipping the code's last character.
    x0, x1 = max(0, layout.code_x0 + 6), min(w, layout.code_x1 + 18)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    processed = _preprocess_for_ocr(crop)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789",
    )
    cleaned = text.replace("\n", " ")
    m = CODE_PATTERN.search(cleaned)
    if not m:
        return None
    raw = m.group(0).replace(" ", "")
    if "_" not in raw:
        # OCR commonly drops the underscore (e.g. "PC3") - reinsert it at the
        # letter/digit boundary so fuzzy-matching against "PC_3" works well.
        letters = re.match(r"[A-Z]+", raw)
        if letters:
            raw = raw[:letters.end()] + "_" + raw[letters.end():]
    return raw


SRNO_PATTERN = re.compile(r"\d{1,2}")


def ocr_row_srno(rgb: np.ndarray, layout: GridLayout, y0: int, y1: int) -> int | None:
    """Reads the Sr.No column's row number directly (e.g. "69" for row 69)
    - a second, independent position signal alongside the Code column.
    Checked against 4 real documents: plain digits read far more reliably
    than the Code column's mixed letters+underscore+digits (which need an
    exact or near-exact fuzzy match against the schema to be useful at
    all) - most rows across every document tested produced a clean,
    sequence-consistent integer, including several rows where the Code
    read was garbled past any usable fuzzy match. Column position is not
    separately calibrated - a generous fixed-width crop immediately left
    of the already-calibrated Code column is enough, since digit OCR
    tolerates extra blank margin far better than it tolerates a tight
    crop clipping a character."""
    h, w, _ = rgb.shape
    x0, x1 = max(0, layout.code_x0 - 65), max(0, layout.code_x0 - 5)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    processed = _preprocess_for_ocr(crop)
    text = pytesseract.image_to_string(
        processed, config="--psm 7 -c tessedit_char_whitelist=0123456789",
    )
    m = SRNO_PATTERN.search(text.strip())
    return int(m.group(0)) if m else None


def fuzzy_match_code(ocr_code: str | None, known_codes: list[str], cutoff: float = 0.75,
                      code_to_position: dict | None = None, cursor: int = 0) -> str | None:
    """Among codes close enough to fuzzy-match, more than one can tie on
    string-similarity ratio - verified on a real failure on BP_15.pdf:
    "OP_4" scored an identical 0.75 against both "DP_4" (position 41,
    exactly where the row actually was) and "PS_4" (position 53), and
    difflib's arbitrary tie-break picked the wrong one, discarding the 12
    real items in between. When `code_to_position` and `cursor` are given,
    ties (and near-ties) are broken by document proximity instead -
    whichever plausible candidate is closest to the current cursor is
    overwhelmingly more likely to be correct than one many positions
    away, since document order is real information a pure string-
    similarity ranking ignores entirely."""
    if not ocr_code:
        return None
    matches = difflib.get_close_matches(ocr_code, known_codes, n=5, cutoff=cutoff)
    if not matches:
        return None
    if not code_to_position:
        return matches[0]
    return min(matches, key=lambda c: abs(code_to_position[c]["position"] - cursor))


# ---------- ink detection (same calibration as the original pipeline) ----------

def _otsu_threshold(values: np.ndarray, bins: int = 100) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.2
    sum_all = np.sum(hist * np.arange(bins))
    sumB = wB = 0.0
    best_var, best_t = -1.0, bins // 2
    for i in range(bins):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB, mF = sumB / wB, (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, i
    return float(edges[best_t])


VALUE_DARK_THRESHOLD = 0.6


def detect_ink_mask(rgb: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    arr = rgb.astype(float) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc, minc = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 0, (maxc - minc) / np.where(maxc == 0, 1, maxc), 0)
    val = maxc
    y0, y1, x0, x1 = region
    region_sat, region_val = sat[y0:y1, x0:x1], val[y0:y1, x0:x1]
    dark = region_val < 0.9
    threshold = 0.2 if dark.sum() < 20 else max(0.12, _otsu_threshold(region_sat[dark]))
    return ((sat > threshold) & (val < 0.9)) | (val < VALUE_DARK_THRESHOLD)


def mark_in_row(rgb: np.ndarray, layout: GridLayout, y0: int, y1: int, min_pixels: int = 80) -> int | None:
    """Returns the 1-5 column of a detected mark within this row band, or
    None if no mark clears min_pixels."""
    h, w, _ = rgb.shape
    region = (max(0, y0), min(h, y1), max(0, layout.answer_left - 10), min(w, layout.answer_right + 10))
    mask = detect_ink_mask(rgb, region)
    ry0, ry1, rx0, rx1 = region
    sub = mask[ry0:ry1, rx0:rx1]
    ys, xs = np.where(sub)
    if len(ys) < min_pixels:
        return None
    cx = float(np.mean(xs)) + rx0
    col_bounds = layout.col_bounds
    for i in range(5):
        hi = col_bounds[i + 1] + (5 if i == 4 else 0)
        if col_bounds[i] <= cx < hi:
            return i + 1
    return None


# ---------- per-half, per-document assembly ----------

@dataclass
class RowResult:
    source: str
    y: float
    value: int | None
    ocr_code: str | None
    matched_code: str | None
    ocr_srno: int | None = None
    anchored_by: str | None = None  # "code", "srno", or None (sequential only)


def process_half(rgb: np.ndarray, source_label: str) -> list[RowResult]:
    layout = find_grid_layout(rgb)
    if layout is None:
        return []
    bands = find_row_bands(rgb, layout)
    results = []
    for y0, y1 in bands:
        ocr_code = ocr_row_code(rgb, layout, y0, y1)
        ocr_srno = ocr_row_srno(rgb, layout, y0, y1)
        value = mark_in_row(rgb, layout, y0, y1)
        results.append(RowResult(source=source_label, y=(y0 + y1) / 2, value=value,
                                  ocr_code=ocr_code, matched_code=None, ocr_srno=ocr_srno))
    return results


@dataclass
class ItemResult:
    item_id: int
    code: str
    value: int | None
    confidence: str
    note: str | None
    source: str | None
    y: float | None


def align_document_to_items(pdf_path: str, items: list, work_dir: str) -> list[ItemResult]:
    """items: schema rows with 'id', 'code', 'position' (1-indexed, document
    order).

    Pass 1 finds "anchors" - rows whose OCR'd code fuzzy-matches a known
    schema code, keeping only matches that move strictly forward in
    document order (rejects a wrong/backwards fuzzy hit). Pass 2 then walks
    the document in segments between consecutive anchors (plus virtual
    start/end anchors at position 0 and n_expected+1) and only assigns rows
    to positions within a segment when the count of unanchored rows found
    there exactly matches the count of schema positions expected there.

    This replaces an earlier version that always filled forward from "the
    last confirmed position + 1" regardless of whether a row had gone
    missing earlier in that stretch. Verified against a real failure on
    BP_15.pdf: a row went undetected a few positions before PC_3/PC_4, so
    the always-fill-forward approach silently attributed PC_3's and PC_4's
    real, correctly-detected marks to the wrong item codes (PC_1/PC_2) -
    the items PC_3/PC_4 then showed as "not detected" even though they
    were read correctly, just under the wrong label. Requiring an exact
    count match per segment means a stretch with a missing row is now
    reported as an honest "can't tell which row is which here" instead of
    confidently assigning data to the wrong items."""
    pages = render_pdf_pages(pdf_path, work_dir)
    code_to_item = {it["code"]: it for it in items}
    known_codes = list(code_to_item.keys())
    n_expected = len(items)

    all_rows: list[RowResult] = []
    for page_path in pages:
        left, right = split_halves(page_path)
        page_name = Path(page_path).name
        if left.shape != (1, 1, 3):
            all_rows.extend(process_half(left, f"{page_name}:L"))
        all_rows.extend(process_half(right, f"{page_name}:R"))

    anchors: list[tuple[int, int]] = []  # (row index into all_rows, schema position)
    # A match is rejected if it would jump the cursor implausibly far
    # ahead. Verified necessary on BP_15.pdf: a 3-character garbled OCR
    # read ("P_3") fuzzy-matched "PS_3" (ratio 0.86, above the 0.75
    # cutoff) when the row's true identity was ~27 positions earlier -
    # short reads are inherently ambiguous against a schema with many
    # codes sharing short suffixes like "_3". No single real gap in this
    # document's row detection has exceeded ~15 positions, so a jump
    # past that is far more likely a coincidental short-string match than
    # a genuine long run of missed rows.
    MAX_PLAUSIBLE_JUMP = 15
    cursor = 0
    for i, row in enumerate(all_rows):
        matched = fuzzy_match_code(row.ocr_code, known_codes, code_to_position=code_to_item, cursor=cursor)
        if matched:
            pos = code_to_item[matched]["position"]
            if cursor < pos <= cursor + MAX_PLAUSIBLE_JUMP:
                row.matched_code = matched
                row.anchored_by = "code"
                anchors.append((i, pos))
                cursor = pos
                continue
        # Code OCR failed or was rejected - fall back to the Sr.No column,
        # a second, independent position signal. Checked across 4 real
        # documents: plain digits read far more reliably than the Code
        # column's mixed letters+underscore+digits, recovering an anchor
        # on several rows where the Code read was too garbled for any
        # fuzzy match at all. Sr.No is the schema position directly (no
        # fuzzy lookup needed) - the same jump-distance cap applies, since
        # a misread digit is just as capable of a coincidentally-plausible
        # wrong jump as a garbled code string.
        if row.ocr_srno and cursor < row.ocr_srno <= cursor + MAX_PLAUSIBLE_JUMP:
            pos = row.ocr_srno
            row.anchored_by = "srno"
            anchors.append((i, pos))
            cursor = pos

    position_to_row: dict[int, RowResult] = {}
    position_to_note: dict[int, str] = {}
    for i, pos in anchors:
        position_to_row[pos] = all_rows[i]

    full_anchors = [(-1, 0)] + anchors + [(len(all_rows), n_expected + 1)]
    for (prev_idx, prev_pos), (next_idx, next_pos) in zip(full_anchors, full_anchors[1:]):
        segment_rows = all_rows[prev_idx + 1:next_idx]
        expected_positions = list(range(prev_pos + 1, next_pos))
        if len(segment_rows) == len(expected_positions):
            for pos, row in zip(expected_positions, segment_rows):
                position_to_row[pos] = row
            continue
        if not expected_positions or not segment_rows:
            continue

        # Counts don't match within this segment - a row was missed, or an
        # extra false-positive band was found, somewhere in this stretch,
        # and which one is unknown. Rather than leaving every position in
        # the segment blank (the earlier, maximally safe behavior), use
        # each row's own OCR reading - even where it was too weak or out
        # of monotonic order to qualify as a hard anchor earlier - as a
        # soft position hint when it lands inside this segment's expected
        # range: a garbled read that still parses to a plausible in-range
        # position is real evidence of where that row belongs, not
        # something to discard just because it didn't independently prove
        # forward progress. Only rows with no usable reading at all are
        # placed by sequential interpolation into whatever positions are
        # still unfilled - every position in a mismatched segment is
        # flagged "low" confidence either way, so a wrong guess here is
        # visible for review rather than presented as a confirmed read.
        soft_placed: dict[int, RowResult] = {}
        unplaced_rows: list[RowResult] = []
        lo, hi = expected_positions[0], expected_positions[-1]
        for row in segment_rows:
            candidate = None
            if row.ocr_srno and lo <= row.ocr_srno <= hi:
                candidate = row.ocr_srno
            else:
                matched = fuzzy_match_code(row.ocr_code, known_codes, code_to_position=code_to_item, cursor=prev_pos)
                if matched:
                    cand_pos = code_to_item[matched]["position"]
                    if lo <= cand_pos <= hi:
                        candidate = cand_pos
            if candidate is not None and candidate not in soft_placed:
                soft_placed[candidate] = row
            else:
                unplaced_rows.append(row)

        note = (
            f"{len(segment_rows)} row(s) found for {len(expected_positions)} expected "
            f"item(s) between schema positions {prev_pos} and {next_pos} - position within "
            "this stretch is not fully certain; please check it manually against the source PDF"
        )
        remaining_positions = [p for p in expected_positions if p not in soft_placed]
        for pos, row in zip(remaining_positions, unplaced_rows):
            soft_placed[pos] = row
        for pos, row in soft_placed.items():
            position_to_row[pos] = row
        for pos in expected_positions:
            position_to_note[pos] = note

    item_results = []
    for it in items:
        pos = it["position"]
        row = position_to_row.get(pos)
        mismatch_note = position_to_note.get(pos)
        if row is None:
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=None, confidence="conflict",
                note=mismatch_note or "no row located for this item in the document",
                source=None, y=None,
            ))
        elif mismatch_note:
            # Placed via best-effort soft positioning within a segment
            # whose row/position counts didn't match - always flagged,
            # whether or not a value was found, since the position itself
            # (not just the mark) is uncertain here.
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=row.value,
                confidence="low" if row.value is not None else "conflict",
                note=mismatch_note, source=row.source, y=row.y,
            ))
        elif row.value is None:
            confidence = "ok" if row.anchored_by else "low"
            if row.anchored_by == "code":
                note = f"row located (code read as '{row.ocr_code}'), but no mark detected in it - please check manually"
            elif row.anchored_by == "srno":
                note = f"row located (Sr.No read as '{row.ocr_srno}'), but no mark detected in it - please check manually"
            else:
                note = "row located by position only (neither code nor Sr.No confidently read) and no mark detected"
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=None, confidence="conflict",
                note=note, source=row.source, y=row.y,
            ))
        else:
            if row.anchored_by:
                confidence, note = "ok", None
            else:
                confidence = "low"
                note = f"row position inferred sequentially (code read as '{row.ocr_code}') - please confirm"
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=row.value, confidence=confidence,
                note=note, source=row.source, y=row.y,
            ))
    return item_results


def crop_source_region(work_dir: str, source: str, y: float, half_width_px: int = 130):
    if not source or y is None:
        return None, "no source/y recorded for this item"
    work_path = Path(work_dir)
    if not work_path.exists():
        return None, f"working directory not found: {work_dir}"
    page_name, _, side = source.partition(":")
    page_path = work_path / page_name
    if not page_path.exists():
        return None, f"page image not found: {page_path}"
    try:
        left, right = split_halves(str(page_path))
        rgb = left if side == "L" else right
        h = rgb.shape[0]
        y0, y1 = max(0, int(y) - half_width_px), min(h, int(y) + half_width_px)
        return Image.fromarray(rgb[y0:y1, :]), None
    except Exception as e:
        return None, f"error reading/cropping {page_path}: {e!r}"
