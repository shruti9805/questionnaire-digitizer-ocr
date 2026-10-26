import tempfile
from pathlib import Path

import streamlit as st

from db import (
    init_db, sync_schema_from_json, list_phases, get_items,
    create_batch, list_batches, create_response, list_responses, get_response,
    save_response_items, get_response_items, update_response_item,
)
from ocr_pipeline import align_document_to_items, crop_source_region, split_halves

st.set_page_config(page_title="Questionnaire Digitizer (OCR)", layout="wide")

init_db()
SCHEMA_DIR = Path(__file__).parent / "schemas"
for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
    sync_schema_from_json(schema_file)

st.title("Questionnaire Digitizer — OCR-anchored")
st.caption(
    "Reads the printed item code next to each row (e.g. \"PC_3\") to tie every "
    "detected - or missing - mark directly to its schema item, instead of "
    "inferring position by counting marks in document order."
)

tab_process, tab_review = st.tabs(["Process & review", "About this approach"])

with tab_process:
    phases = list_phases()
    if not phases:
        st.warning("No schema found in schemas/. Add a phase_*.json file and restart.")
        st.stop()
    phase_options = {p["name"]: p["id"] for p in phases}
    phase_label = st.selectbox("Phase", list(phase_options.keys()))
    phase_id = phase_options[phase_label]
    phase_items = get_items(phase_id)

    st.markdown("### Upload a new batch of scans")
    batch_label = st.text_input("Batch label (optional)")
    uploads = st.file_uploader("Upload scanned PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Process uploaded PDFs"):
        batch_id = create_batch(phase_id, batch_label or None)
        results_summary = []
        for pdf_upload in uploads:
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / pdf_upload.name
                pdf_path.write_bytes(pdf_upload.getbuffer())
                response_dir = Path(tempfile.mkdtemp(prefix="response_"))
                with st.spinner(f"Reading {pdf_upload.name}..."):
                    item_results = align_document_to_items(str(pdf_path), phase_items, str(response_dir))
                response_id = create_response(
                    batch_id, source_pdf_filename=pdf_upload.name, pages_dir=str(response_dir),
                    respondent_label=Path(pdf_upload.name).stem,
                )
                save_response_items(response_id, item_results)
                n_flagged = sum(1 for r in item_results if r.confidence != "ok")
                n_missing = sum(1 for r in item_results if r.value is None)
                results_summary.append({
                    "file": pdf_upload.name,
                    "response": f"#{response_id}",
                    "flagged": f"{n_flagged} of {len(item_results)}",
                    "no value": n_missing,
                })
        st.success(f"Processed {len(uploads)} file(s) into batch #{batch_id}.")
        st.dataframe(results_summary, hide_index=True, width="stretch")

    st.markdown("### Review a response")
    batches = list_batches(phase_id)
    if not batches:
        st.write("No batches yet for this phase.")
    else:
        batch_options = {f"#{b['id']} {b['label'] or ''} ({b['uploaded_at'][:10]})": b["id"] for b in batches}
        batch_label_sel = st.selectbox("Batch", list(batch_options.keys()))
        batch_id = batch_options[batch_label_sel]
        responses = list_responses(batch_id)
        if not responses:
            st.write("No responses in this batch.")
        else:
            response_options = {
                f"#{r['id']} {r['respondent_label'] or r['source_pdf_filename']}": r["id"] for r in responses
            }
            response_label = st.selectbox("Response", list(response_options.keys()))
            response_id = response_options[response_label]
            response_row = get_response(response_id)
            response_items = get_response_items(response_id)

            if any("no row located" in (ri["note"] or "") for ri in response_items):
                st.warning(
                    "Some rows could not be located in this document at all (table "
                    "structure wasn't detected there) - those items have no "
                    "automated value and need manual entry from the source PDF."
                )

            st.markdown("#### Items needing review")
            flagged = [ri for ri in response_items if ri["confidence"] != "ok"]
            if not flagged:
                st.write("Nothing flagged — every item was read with high confidence.")
            NOT_DETECTED = "— not detected, choose a value —"
            for ri in flagged:
                with st.container(border=True):
                    st.markdown(f"**{ri['code']}** — {ri['statement_en']}")
                    st.caption(f"{ri['confidence']}: {ri['note']}")
                    crop, crop_error = crop_source_region(response_row["pages_dir"], ri["source"], ri["y"])
                    if crop is not None:
                        st.image(crop, width=900)
                    else:
                        st.warning(f"Source image unavailable: {crop_error}")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        options = [NOT_DETECTED, 1, 2, 3, 4, 5] if ri["value"] is None else [1, 2, 3, 4, 5]
                        corrected = st.selectbox(
                            "Value", options,
                            index=0 if ri["value"] is None else ri["value"] - 1,
                            key=f"correct_{ri['id']}",
                        )
                    with col2:
                        st.write("")
                        if st.button("Confirm this value", key=f"confirmbtn_{ri['id']}"):
                            if corrected == NOT_DETECTED:
                                st.error("Choose an actual value before confirming - this item had no detected mark.")
                            else:
                                update_response_item(ri["id"], corrected, confidence="ok")
                                st.success(f"{ri['code']} confirmed as {corrected}.")
                                st.rerun()

            st.markdown("#### Full response")
            st.dataframe(
                [{"code": ri["code"], "value": ri["value"], "confidence": ri["confidence"]} for ri in response_items],
                hide_index=True, width="stretch",
            )
            n_unresolved = sum(1 for ri in response_items if ri["confidence"] != "ok")
            if n_unresolved:
                st.warning(f"{n_unresolved} item(s) above are still flagged — export will include them as-is.")

with tab_review:
    st.markdown("""
Each row's printed item code (e.g. "PC_3") is read directly via OCR and
matched against the schema's known code list, so a detected - or missing -
mark is tied to its real schema position before any sequential counting
happens. When OCR can't read a row's code confidently, that row's position
is inferred by continuing from the last OCR-confirmed position (the same
assumption the sequential approach always made, just re-anchored much more
often instead of never).

**Current limitations (first version):**
- Row and column boundary detection on real scans is imperfect - some rows
  are still missed entirely ("no row located"), and OCR doesn't confidently
  read every row's code even when the row itself is found correctly.
- This was built and tested against one real respondent's scan (BP_15.pdf)
  from the pilot batch; it has not yet been validated across the full
  range of scan quality seen in that batch.
- Treat items marked "low" or "conflict" confidence as needing the same
  manual verification against the source PDF as the original tool's
  flagged items.
""")
