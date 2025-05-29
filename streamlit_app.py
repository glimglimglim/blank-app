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

import openai
import streamlit as st
from PIL import Image

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

# Expected JSON fields and system prompt
DL_FIELDS = [
    "license_number","class","first_name","middle_name","last_name",
    "address","city","state","zip","date_of_birth","issue_date",
    "expiration_date","sex","eye_color","height","organ_donor",
]

SYSTEM_PROMPT = (
    "You are an identity-document data extractor. "
    "Extract the following fields from a U.S. driver's-license image and return *only* valid JSON "
    "with exactly these keys in this order: "
    + ", ".join(DL_FIELDS)
    + ". Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — API Key Management
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔑 API Keys")

    # OpenAI key
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not openai.api_key:
        key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if key:
            openai.api_key = key
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
    contents = [
        types.Content(
            role="system",
            parts=[types.Part.from_text(SYSTEM_PROMPT)]
        )
    ] + [
        types.Content(
            role="user",
            parts=[types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")]
        )
        for b64 in b64_images
    ]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose an image or PDF of a driver's license",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded_file and openai.api_key and gemini_key:
    if st.button("🚀 Extract with GPT-4o & Gemini", type="primary"):
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

        st.success("Extraction complete!")

        col_img, col_oai, col_gem = st.columns(3)

        with col_img:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        with col_oai:
            st.subheader("🤖 GPT-4o-mini JSON")
            st.json(dl_openai, expanded=True)
            st.download_button(
                "💾 Download GPT-4o JSON",
                data=json.dumps(dl_openai, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_gpt4o.json",
                mime="application/json",
            )

        with col_gem:
            st.subheader("🤖 Gemini JSON")
            st.json(dl_gemini, expanded=True)
            st.download_button(
                "💾 Download Gemini JSON",
                data=json.dumps(dl_gemini, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_gemini.json",
                mime="application/json",
            )

elif uploaded_file:
    st.info("Please provide both OpenAI and Gemini API keys in the sidebar to proceed.")
else:
    st.write("👈 Upload a file and provide API keys to get started.")
