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
    two immediately left of the answer grid's left edge."""
    h, w, _ = rgb.shape
    gray = np.array(Image.fromarray(rgb).convert("L"))
    dark = gray < 235

    x0 = int(w * 0.5)
    grid_lines = _find_lines(dark, h, w, min_run_frac, (x0, w))
    if not grid_lines:
        return None
    xs_only = sorted(l[0] for l in grid_lines)
    left, right = xs_only[0], xs_only[-1]
    grid_width = right - left
    if not (0.12 * w <= grid_width <= 0.26 * w):
        return None
    col_bounds = xs_only if len(xs_only) == 6 else [left + (right - left) * i / 5 for i in range(6)]

    # Vertical lines left of the grid (Sr.No|Code and Code|Statement
    # boundaries), restricted to the table's own y-span (not the whole
    # half-page, since unrelated tall marks elsewhere on the page -
    # signatures, demographic-section boxes - would otherwise register as
    # false column lines). Found by column-wise dark-pixel density rather
    # than a contiguous-run test: the same thin/broken-line issue found
    # with the horizontal row rules (see find_row_bands) applies here too
    # - a genuine column boundary's darkness isn't always one unbroken
    # run top-to-bottom, so density + a thinness filter is more robust.
    # When fewer than 2 candidate lines are found this way, fall back to a
    # proportion of `left` calibrated against several real, correctly
    # line-detected halves of BP_15.pdf (ratios of 0.15-0.22 and
    # 0.25-0.31) - less precise, but better than nothing.
    MARGIN = 150
    y_top = max(0, min(l[1] for l in grid_lines) - MARGIN)
    y_bottom = max(l[2] for l in grid_lines) + MARGIN
    dark_table_rows = dark[y_top:y_bottom, :max(0, left - 5)]
    col_dark_frac = dark_table_rows.mean(axis=0)
    cutoff = max(0.35, _otsu_threshold(col_dark_frac, bins=50)) if col_dark_frac.max() >= 0.05 else 1.0
    is_line = col_dark_frac > cutoff
    MAX_LINE_THICKNESS = 6
    all_xs = []
    cur = []
    for i, v in enumerate(is_line):
        if v:
            cur.append(i)
        elif cur:
            if len(cur) <= MAX_LINE_THICKNESS:
                all_xs.append(int(np.mean(cur)))
            cur = []
    if cur and len(cur) <= MAX_LINE_THICKNESS:
        all_xs.append(int(np.mean(cur)))
    all_xs.sort()
    if len(all_xs) >= 2 and (all_xs[-1] - all_xs[-2]) >= 0.08 * left:
        code_x1, code_x0 = all_xs[-1], all_xs[-2]
    else:
        code_x0, code_x1 = int(0.14 * left), int(0.30 * left)

    return GridLayout(answer_left=left, answer_right=right, col_bounds=col_bounds,
                       code_x0=code_x0, code_x1=code_x1, y_top=y_top, y_bottom=y_bottom)


def find_row_bands(rgb: np.ndarray, layout: GridLayout) -> list[tuple[int, int]]:
    """Horizontal rule lines bounding each table row.

    Unlike the bold, solidly-printed vertical double-lines framing the
    answer grid, the horizontal row rules in a real scan (verified against
    BP_15.pdf) are thin and broken by scan noise/antialiasing - the
    longest *contiguous* dark run at a true rule-line y was only ~27% of
    the table width, well short of what a contiguous-run test (as used for
    the vertical lines) would accept. A plain statement-text row, by
    contrast, has a sustained ~20-26% dark-pixel *density* with no sharp
    peak. The actual rule lines stood out clearly as a density peak
    (~45-52%) against that text baseline, so row boundaries are found via
    per-row dark-pixel density (not contiguous-run length), separated from
    the text baseline with the same per-document Otsu split already used
    to calibrate ink detection - a fixed density cutoff would face the
    same per-document miscalibration risk demonstrated earlier when a
    single fixed pixel-count threshold was tried for ink detection."""
    h, w, _ = rgb.shape
    gray = np.array(Image.fromarray(rgb).convert("L"))
    # Restricted to the Sr.No/Code columns only (not into the Statement
    # column): that column holds only Latin digits/codes, while the
    # Statement column's Hindi text carries a continuous horizontal
    # headstroke (shirorekha) above each word that reads as a dense,
    # thin, wide dark band almost indistinguishable from a true rule line
    # when scanned across the full table width.
    x0, x1 = max(0, layout.code_x0 - 10), min(w, layout.code_x1)
    sub = gray[layout.y_top:layout.y_bottom, x0:x1]
    dark_frac = (sub < 245).mean(axis=1)
    if dark_frac.max() < 0.05:
        return []
    cutoff = max(0.35, _otsu_threshold(dark_frac, bins=50))
    is_line = dark_frac > cutoff

    # A density peak alone doesn't distinguish a true rule line from a
    # densely-printed text line (e.g. bolder Hindi script sometimes also
    # crosses the density cutoff) - but a rule line is geometrically thin
    # (a few px tall) while a text line spans a full glyph height, so
    # filter candidate groups by thickness too.
    MAX_LINE_THICKNESS = 6
    groups = []
    cur = []
    for i, v in enumerate(is_line):
        if v:
            cur.append(i)
        elif cur:
            if len(cur) <= MAX_LINE_THICKNESS:
                groups.append(int(np.mean(cur)) + layout.y_top)
            cur = []
    if cur and len(cur) <= MAX_LINE_THICKNESS:
        groups.append(int(np.mean(cur)) + layout.y_top)

    bands = []
    for y_top, y_bot in zip(groups, groups[1:]):
        if y_bot - y_top > 20:  # skip degenerate adjacent-line duplicates
            bands.append((y_top, y_bot))
    return bands


# ---------- OCR ----------

CODE_PATTERN = re.compile(r"[A-Z]{1,3}_?\s?\d{1,2}")


def _preprocess_for_ocr(crop: Image.Image) -> Image.Image:
    gray = crop.convert("L")
    scale = 3
    up = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    arr = np.array(up)
    bw = ((arr > 150) * 255).astype(np.uint8)
    return Image.fromarray(bw)


def ocr_row_code(rgb: np.ndarray, layout: GridLayout, y0: int, y1: int) -> str | None:
    h, w, _ = rgb.shape
    x0, x1 = max(0, layout.code_x0), min(w, layout.code_x1)
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


def fuzzy_match_code(ocr_code: str | None, known_codes: list[str], cutoff: float = 0.75) -> str | None:
    if not ocr_code:
        return None
    matches = difflib.get_close_matches(ocr_code, known_codes, n=1, cutoff=cutoff)
    return matches[0] if matches else None


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


def process_half(rgb: np.ndarray, source_label: str) -> list[RowResult]:
    layout = find_grid_layout(rgb)
    if layout is None:
        return []
    bands = find_row_bands(rgb, layout)
    results = []
    for y0, y1 in bands:
        ocr_code = ocr_row_code(rgb, layout, y0, y1)
        value = mark_in_row(rgb, layout, y0, y1)
        results.append(RowResult(source=source_label, y=(y0 + y1) / 2, value=value,
                                  ocr_code=ocr_code, matched_code=None))
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
    order). Each detected row is matched to a schema code via OCR when
    possible; otherwise it continues from the last OCR-confirmed position,
    the same sequential assumption the original pipeline always used, but
    now re-anchored at nearly every row instead of never."""
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

    position_to_row: dict[int, RowResult] = {}
    cursor = 0  # last confirmed schema position (1-indexed); next unanchored row = cursor + 1
    for row in all_rows:
        matched = fuzzy_match_code(row.ocr_code, known_codes)
        if matched:
            pos = code_to_item[matched]["position"]
            if pos >= cursor:  # reject a match that would move backwards - likely a false OCR hit
                row.matched_code = matched
                cursor = pos
                if pos not in position_to_row:
                    position_to_row[pos] = row
                continue
        # no usable OCR anchor for this row: fall back to sequential continuation
        pos = cursor + 1
        if pos <= n_expected and pos not in position_to_row:
            position_to_row[pos] = row
            cursor = pos

    item_results = []
    for it in items:
        pos = it["position"]
        row = position_to_row.get(pos)
        if row is None:
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=None, confidence="conflict",
                note="no row located for this item in the document", source=None, y=None,
            ))
        elif row.value is None:
            confidence = "ok" if row.matched_code == it["code"] else "low"
            note = (
                "row located (code read as "
                f"'{row.ocr_code}'), but no mark detected in it - please check manually"
                if row.matched_code
                else "row located by position only (code not confidently read) and no mark detected"
            )
            item_results.append(ItemResult(
                item_id=it["id"], code=it["code"], value=None, confidence="conflict",
                note=note, source=row.source, y=row.y,
            ))
        else:
            if row.matched_code == it["code"]:
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
