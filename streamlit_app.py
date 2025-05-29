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

# Optional PDF support
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# Import and configure Google's Gemini
from google import genai


# Configure API keys
st.set_page_config(page_title="Driver‑License Extractor", layout="centered")
st.title("🪪 ➜ 📋  Driver‑License Data Extractor")

with st.sidebar:
    st.header("🔑 API Keys")
    # OpenAI key
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not openai.api_key:
        api_key_in = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if api_key_in:
            openai.api_key = api_key_in
    else:
        st.success("OpenAI key loaded.")

    # Gemini key
    gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    if gemini_key:
        genai.configure(api_key=gemini_key)
        st.success("Gemini key loaded.")
    else:
        gemini_key = st.text_input("Gemini API key", type="password", placeholder="...", help="Required for Gemini extraction")
        if gemini_key:
            genai.configure(api_key=gemini_key)

    st.markdown("---\n⚠️ **Privacy reminder:** ensure you are authorised to process any personal data you upload.")

# Expected JSON fields
DL_FIELDS = [
    "license_number","class","first_name","middle_name","last_name",
    "address","city","state","zip","date_of_birth","issue_date",
    "expiration_date","sex","eye_color","height","organ_donor",
]

SYSTEM_PROMPT = (
    "You are an identity-document data extractor. "
    "Extract the following fields from a U.S. driver's-license image and return *only* valid JSON" \
    " with exactly these keys in this order: " + ", ".join(DL_FIELDS) + ". "
    "Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# Utility: image conversion

def _file_to_images(path: Path) -> List[Image.Image]:
    """Convert PDF or image path into list of PIL Images."""
    mime, _ = mimetypes.guess_type(path)
    if mime == "application/pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image not installed or Poppler missing.")
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

# Model calls

def gpt4o_dl_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
            for b64 in b64_images
        ]},
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
            for b64 in b64_images
        ]},
    ]
    # Adjust model name as needed (e.g., "gemini-1.0" or "gemini-pro")
    resp = genai.chat.completions.create(
        model="gemini-pro",
        messages=messages,
        temperature=0.0,
        max_output_tokens=4096,
    )
    # Assuming the content is valid JSON string
    return json.loads(resp.choices[0].message.content)

# Streamlit Uploader
uploaded_file = st.file_uploader(
    "Choose an image or PDF of a driver's license",
    type=["pdf","png","jpg","jpeg","tiff","tif"],
)

# Proceed only if file and both keys are provided
if uploaded_file and openai.api_key and gemini_key:
    if st.button("🚀 Extract with GPT-4o & Gemini", type="primary"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        # Convert file
        with st.spinner("Converting file …"):
            try:
                b64_chunks = file_to_base64_chunks(tmp_path)
                images = _file_to_images(tmp_path)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        # Call both models
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

        # Layout: images + two results
        col_img, col_oai, col_gem = st.columns(3)

        with col_img:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        # OpenAI results
        with col_oai:
            st.subheader("🤖 GPT-4o-mini JSON")
            st.json(dl_openai, expanded=True)
            st.download_button(
                label="💾 Download GPT-4o JSON",
                data=json.dumps(dl_openai, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_gpt4o.json",
                mime="application/json",
            )

        # Gemini results
        with col_gem:
            st.subheader("🤖 Gemini JSON")
            st.json(dl_gemini, expanded=True)
            st.download_button(
                label="💾 Download Gemini JSON",
                data=json.dumps(dl_gemini, indent=2),
                file_name=f"{Path(uploaded_file.name).stem}_gemini.json",
                mime="application/json",
            )

elif uploaded_file and (not openai.api_key or not gemini_key):
    st.info("Please provide both OpenAI and Gemini API keys in the sidebar to proceed.")
else:
    st.write("👈 Upload a file and provide API keys to get started.")
