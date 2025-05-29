from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List

import boto3
import openai
import streamlit as st
from PIL import Image

# Optional PDF support
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Driver-License Extractor", layout="centered")
st.title("🪪 ➜ 📋  Driver-License Data Extractor")

DL_FIELDS = [
    "license_number", "class", "first_name", "middle_name", "last_name",
    "address", "city", "state", "zip", "date_of_birth", "issue_date",
    "expiration_date", "sex", "eye_color", "height", "organ_donor",
]

SYSTEM_PROMPT = (
    "You are an identity-document data extractor. "
    "Extract the following fields from a U.S. driver's-license image and return *only* valid JSON "
    "with exactly these keys in this order: "
    + ", ".join(DL_FIELDS)
    + ". Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — API Key & Client Initialization
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔑 API Keys & Clients")

    # OpenAI key
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not openai.api_key:
        k = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if k:
            openai.api_key = k
    else:
        st.success("OpenAI key loaded.")

    # Gemini key & client
    gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    if not gemini_key:
        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="…",
            help="Required for Gemini extraction"
        )
    if gemini_key:
        client = genai.Client(api_key=gemini_key)
        st.success("Gemini client initialized.")

    # AWS credentials & Textract client from secrets.toml
    try:
        aws_cfg = st.secrets["aws"]
        textract = boto3.client(
            "textract",
            aws_access_key_id=aws_cfg["aws_access_key_id"],
            aws_secret_access_key=aws_cfg["aws_secret_access_key"],
            aws_session_token=aws_cfg["aws_session_token"],
            region_name=aws_cfg.get("region_name", "us-east-1"),
        )
        st.success("AWS Textract client initialized.")
    except Exception:
        st.error("Make sure you have an [aws] section in .streamlit/secrets.toml")

    st.markdown(
        "---\n"
        "⚠️ **Privacy reminder:** ensure you are authorised to process any personal data you upload."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def _file_to_images(path: Path) -> List[Image.Image]:
    mime, _ = mimetypes.guess_type(path)
    if mime == "application/pdf":
        if convert_from_path is None:
            raise RuntimeError("Install pdf2image and Poppler for PDF support.")
        return convert_from_path(path, dpi=300)
    if mime and mime.startswith("image/"):
        return [Image.open(path)]
    raise ValueError(f"Unsupported file type: {path}")

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def file_to_base64_chunks(path: Path) -> List[str]:
    return [_pil_to_base64(im.convert("RGB")) for im in _file_to_images(path)]

def render_fields(container, title: str, data: dict):
    """Display each DL_FIELD in a neat two-column layout under a given container."""
    container.subheader(title)
    for field in DL_FIELDS:
        label = field.replace("_", " ").title()
        value = data.get(field, "") or "—"
        a, b = container.columns([1, 3])
        a.markdown(f"**{label}**")
        b.markdown(value)

# ──────────────────────────────────────────────────────────────────────────────
# Model Invocation Functions
# ──────────────────────────────────────────────────────────────────────────────

def gpt4o_dl_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
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

def gemini_dl_from_images(b64_images: List[str]) -> dict:
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
        for b64 in b64_images
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=4096
        ),
        contents=image_parts
    )
    return json.loads(response.text)

def textract_dl_from_images(path: Path) -> dict:
    """
    Extracts key-value pairs from driver's license images using AWS Textract.
    Returns a dict with exactly the DL_FIELDS keys.
    """
    FIELD_KEYWORDS = {
        "license_number": ["license", "lic no", "dl number"],
        "class": ["class"],
        "first_name": ["first name", "given name"],
        "middle_name": ["middle name"],
        "last_name": ["last name", "surname"],
        "address": ["address"],
        "city": ["city"],
        "state": ["state"],
        "zip": ["zip", "postal code"],
        "date_of_birth": ["date of birth", "dob"],
        "issue_date": ["date of issue", "issue date"],
        "expiration_date": ["expiration date", "exp date", "exp"],
        "sex": ["sex", "gender"],
        "eye_color": ["eye color", "eyes"],
        "height": ["height"],
        "organ_donor": ["organ donor"],
    }
    results = {k: "" for k in DL_FIELDS}

    images = _file_to_images(path)
    for img in images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        resp = textract.analyze_document(Document={'Bytes': buf.getvalue()}, FeatureTypes=['FORMS'])
        blocks = resp.get('Blocks', [])

        # Build block maps
        block_map = {b['Id']: b for b in blocks}
        key_map = {
            b['Id']: b for b in blocks
            if b['BlockType']=='KEY_VALUE_SET' and 'KEY' in b.get('EntityTypes', [])
        }
        value_map = {
            b['Id']: b for b in blocks
            if b['BlockType']=='KEY_VALUE_SET' and 'VALUE' in b.get('EntityTypes', [])
        }

        def get_text(block):
            text = ""
            for rel in block.get('Relationships', []):
                if rel['Type']=='CHILD':
                    for cid in rel['Ids']:
                        word = block_map.get(cid)
                        if word and word['BlockType']=='WORD':
                            text += word['Text'] + ' '
            return text.strip()

        # Extract key-values
        kvs: dict[str, str] = {}
        for key_id, key_block in key_map.items():
            key_text = get_text(key_block).lower()
            val_text = ""
            for rel in key_block.get('Relationships', []):
                if rel['Type']=='VALUE':
                    for vid in rel['Ids']:
                        val_block = value_map.get(vid)
                        if val_block:
                            val_text = get_text(val_block)
            kvs[key_text] = val_text

        # Map to DL_FIELDS
        for field, keywords in FIELD_KEYWORDS.items():
            for key_text, val_text in kvs.items():
                if any(keyword in key_text for keyword in keywords):
                    results[field] = val_text
                    break

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose an image or PDF of a driver's license",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded_file and openai.api_key and gemini_key:
    if st.button("🚀 Extract with GPT-4o, Gemini & AWS Textract", type="primary"):
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

        with st.spinner("Extracting with GPT-4o-mini …"):
            try:
                dl_openai = gpt4o_dl_from_images(b64_chunks)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                dl_openai = {k: "" for k in DL_FIELDS}

        with st.spinner("Extracting with Gemini …"):
            try:
                dl_gemini = gemini_dl_from_images(b64_chunks)
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                dl_gemini = {k: "" for k in DL_FIELDS}

        with st.spinner("Extracting with AWS Textract …"):
            try:
                dl_textract = textract_dl_from_images(tmp_path)
            except Exception as e:
                st.error(f"AWS Textract error: {e}")
                dl_textract = {k: "" for k in DL_FIELDS}

        st.success("Extraction complete!")

        # two columns: image on left, model outputs on right in tabs
        col_img, col_models = st.columns([1, 2], gap="large")

        with col_img:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        with col_models:
            tabs = st.tabs(["🤖 OpenAI", "🤖 Gemini", "🧾 Textract"])
            for tab, title, data in zip(
                tabs,
                ["GPT-4o-mini Fields", "Gemini Fields", "Textract Fields"],
                [dl_openai, dl_gemini, dl_textract],
            ):
                with tab:
                    render_fields(tab, title, data)

elif uploaded_file:
    st.info("Please provide OpenAI, Gemini, and AWS credentials to proceed.")
else:
    st.write("👈 Upload a file and provide API keys to get started.")
