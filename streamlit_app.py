"""
Streamlit app to extract structured JSON from U.S. driver's‑license images and preview a pre‑filled form.

Install dependencies:
    pip install streamlit openai pillow pdf2image

`pdf2image` requires the Poppler utilities (https://poppler.freedesktop.org) on your PATH.

Run the app **from a terminal** with:

    streamlit run streamlit_driver_license_extractor.py

Running it via plain `python streamlit_driver_license_extractor.py` will not spin‑up the Streamlit
server and will show *ScriptRunContext* warnings.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import openai
import streamlit as st
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:  # Optional dependency.
    convert_from_path = None

###############################################################################
# Configuration
###############################################################################

# The JSON schema we expect back from the model.
DL_FIELDS = [
    "license_number",
    "class",
    "first_name",
    "middle_name",
    "last_name",
    "address",
    "city",
    "state",
    "zip",
    "date_of_birth",
    "issue_date",
    "expiration_date",
    "sex",
    "eye_color",
    "height",
    "organ_donor",
]

SYSTEM_PROMPT: str = (
    "You are an identity‑document data extractor. "
    "Extract the following fields from a U.S. driver's‑license image and return *only* **valid JSON** with exactly these keys, in this order: "
    f"{', '.join(DL_FIELDS)}. "
    "Use ISO‑8601 dates (YYYY‑MM‑DD). If a field is missing, set its value to an empty string. "
    "Do **not** output any other keys or explanatory text."
)

###############################################################################
# Utility functions
###############################################################################

def _file_to_images(path: Path) -> List[Image.Image]:
    """Convert a PDF (all pages) or a single image file into a list of PIL Images."""
    mime, _ = mimetypes.guess_type(path)

    if mime == "application/pdf":
        if convert_from_path is None:
            raise RuntimeError(
                "`pdf2image` isn't installed, or Poppler is missing. "
                "Install with `pip install pdf2image` and add Poppler utilities to PATH."
            )
        # 300 DPI balances OCR accuracy and file size.
        return convert_from_path(path, dpi=300)

    if mime and mime.startswith("image/"):
        return [Image.open(path)]

    raise ValueError(f"Unsupported file type: {path}")


def _pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base‑64‑encoded PNG string (without the prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def file_to_base64_chunks(path: Path) -> List[str]:
    """Return one base‑64 PNG string *per* page / image in the file."""
    return [_pil_to_base64(im.convert("RGB")) for im in _file_to_images(path)]


def gpt4o_dl_from_images(b64_images: List[str]) -> dict:
    """Call GPT‑4o‑mini with the images and get the driver's‑license JSON back as a Python dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
                for b64 in b64_images
            ],
        },
    ]

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )

    return json.loads(resp.choices[0].message.content)


def _safe_date(value: str) -> Optional[date]:
    """Attempt to parse YYYY‑MM‑DD date strings; return None on failure."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(page_title="Driver‑License Extractor", layout="centered")
st.title("🪪 ➜ 📋  Driver‑License Data Extractor")

st.markdown(
    "Upload a driver's‑license photo (or PDF) and receive structured JSON **plus** an interactive, pre‑filled form. "
    "No data is stored server‑side."
)

with st.sidebar:
    st.header("🔑 OpenAI API Key")

    # Automatically use the key from secrets or environment variable, no manual input
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY", "")

    if not openai.api_key:
        st.warning("⚠️ No OpenAI API key found in environment or Streamlit secrets.")
    
    st.markdown(
        "---\n⚠️ **Privacy reminder:** ensure you are authorised to process any personal data you upload."
    )

        "---\n⚠️ **Privacy reminder:** ensure you are authorised to process any personal data you upload."
    )

uploaded_file = st.file_uploader(
    "Choose an image or PDF of a driver's license",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

# A small stateful flag so the preview checkbox only shows after a successful run.
show_images = st.session_state.get("show_images", False)

if uploaded_file and (openai.api_key or api_key_input):
    if st.button("🚀 Extract License Data", type="primary"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Converting file …"):
            try:
                b64_chunks = file_to_base64_chunks(tmp_path)
                images = _file_to_images(tmp_path)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        with st.spinner(f"Sending {len(b64_chunks)} page(s)/image(s) to GPT‑4o‑mini …"):
            try:
                dl_json = gpt4o_dl_from_images(b64_chunks)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

        st.success("Extraction complete!")

        # Side‑by‑side layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        with col2:
            st.subheader("🧾 Extracted JSON")
            st.json(dl_json, expanded=True)
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(dl_json, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_license.json",
                mime="application/json",
            )

            st.divider()
            st.subheader("📋 Driver‑License Form")
            with st.form("dl_form"):
                lic_num = st.text_input("License Number", dl_json.get("license_number", ""))
                cls = st.text_input("Class", dl_json.get("class", ""))
                first = st.text_input("First Name", dl_json.get("first_name", ""))
                middle = st.text_input("Middle Name", dl_json.get("middle_name", ""))
                last = st.text_input("Last Name", dl_json.get("last_name", ""))
                address = st.text_input("Address", dl_json.get("address", ""))
                city = st.text_input("City", dl_json.get("city", ""))
                state_val = st.text_input("State", dl_json.get("state", ""))
                zip_code = st.text_input("ZIP", dl_json.get("zip", ""))

                dob_raw = dl_json.get("date_of_birth", "")
                dob = _safe_date(dob_raw) or date.today()
                dob_in = st.date_input("Date of Birth", dob)

                iss_raw = dl_json.get("issue_date", "")
                iss = _safe_date(iss_raw) or date.today()
                iss_in = st.date_input("Issue Date", iss)

                exp_raw = dl_json.get("expiration_date", "")
                exp = _safe_date(exp_raw) or date.today()
                exp_in = st.date_input("Expiration Date", exp)

                sex = st.text_input("Sex", dl_json.get("sex", ""))
                eye = st.text_input("Eye Color", dl_json.get("eye_color", ""))
                height = st.text_input("Height", dl_json.get("height", ""))
                organ = st.selectbox("Organ Donor", ["", "Yes", "No"], index=["", "Yes", "No"].index(dl_json.get("organ_donor", "")))

                submitted = st.form_submit_button("✅ Save / Update")
                if submitted:
                    st.success("Form submitted (not persisted in this demo).")

            st.caption("All form data remains in the browser session and is **not** transmitted.")
