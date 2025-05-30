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
from langfuse.decorators import observe
from langfuse.openai import openai  # Langfuse-wrapped OpenAI client
import streamlit as st
from PIL import Image

# Optional PDF support
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# ──────────────────────────────────────────────────────────────────────────────
# Langfuse Configuration (via environment variables)
# ──────────────────────────────────────────────────────────────────────────────
LANGFUSE_SECRET_KEY = "sk-lf-884f8f3a-6fcb-41a0-831a-018b355a03b4"
LANGFUSE_PUBLIC_KEY = "pk-lf-9b6ba0a4-31cd-4347-ab73-17d0c35786c"
LANGFUSE_HOST = "https://langfuse.ai.wrs.dev"

os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit App Configuration (must be first Streamlit command)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document Extractor", layout="centered")

# ──────────────────────────────────────────────────────────────────────────────
# Document Type Definitions
# ──────────────────────────────────────────────────────────────────────────────
DOCUMENT_TYPES = {
    "Driver's License": {
        "fields": [
            "license_number", "class", "first_name", "middle_name", "last_name",
            "address", "city", "state", "zip", "date_of_birth", "issue_date",
            "expiration_date", "sex", "eye_color", "hair", "height", "organ_donor", "weight"
        ],
        "title": "🪪 ➜ 📋  Driver-License Data Extractor"
    },
    "Insurance Card": {
        "fields": [
            "insurance_company", "member_id", "group_number", "plan_type",
            "insured_name", "insured_dob", "relationship", "effective_date",
            "expiration_date", "copay", "rx_bin", "rx_pcn", "customer_service_number"
        ],
        "title": "💳 ➜ 📋  Insurance Card Data Extractor"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI — Document Type Selection and Title
# ──────────────────────────────────────────────────────────────────────────────
document_type = st.selectbox("Select document type", list(DOCUMENT_TYPES.keys()))
st.title(DOCUMENT_TYPES[document_type]["title"])
FIELDS = DOCUMENT_TYPES[document_type]["fields"]

# ──────────────────────────────────────────────────────────────────────────────
# Prompt for LLMs
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    f"You are an identity-document data extractor. "
    f"Extract the following fields from a U.S. {document_type.lower()} image and return *only* valid JSON "
    f"with exactly these keys in this order: " + ", ".join(FIELDS) +
    ". Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — API Key & Client Initialization
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys & Clients")

    # OpenAI (Langfuse-wrapped)
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not openai.api_key:
        k = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if k:
            openai.api_key = k
    else:
        st.success("OpenAI key loaded.")

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    if not gemini_key:
        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="…",
            help="Required for Gemini extraction"
        )
    if gemini_key:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)
        st.success("Gemini client initialized.")

    # AWS Textract
    try:
        aws_cfg = st.secrets["aws"]
        textract = boto3.client(
            "textract",
            aws_access_key_id=aws_cfg["aws_access_key_id"],
            aws_secret_access_key=aws_cfg["aws_secret_access_key"],
            aws_session_token=aws_cfg.get("aws_session_token"),
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


def render_fields_grid(container, title: str, data: dict, num_cols: int = 3):
    container.subheader(title)
    cols = container.columns(num_cols)
    for idx, field in enumerate(FIELDS):
        col = cols[idx % num_cols]
        label = field.replace("_", " ").title()
        value = data.get(field, "") or ""
        col.markdown(f"""
            <div style="display:flex; flex-direction:column; gap:4px;">
                <div style="font-weight:bold;">{label}</div>
                <div style="
                    background-color:#f8f9fa;
                    padding:8px;
                    border-radius:6px;
                    color:#111;
                    min-height:38px;
                    font-family:monospace;
                    font-size:0.95em;
                    word-wrap:break-word;
                    border: 1px solid #ddd;
                ">{value}</div>
            </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model Invocation Functions with Langfuse Tracing
# ──────────────────────────────────────────────────────────────────────────────

@observe(as_type="generation", name="GPT-4.1-mini Extraction")
def model_dl_from_images(b64_images: List[str], model_name: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
            for b64 in b64_images
        ]},
    ]
    resp = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )
    return json.loads(resp.choices[0].message.content)

@observe(as_type="custom", name="AWS Textract Extraction")
def textract_from_images(path: Path) -> dict:
    # Only supported for Driver's License; returns empty for Insurance
    results = {k: "" for k in FIELDS}
    if document_type == "Driver's License":
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
        images = _file_to_images(path)
        for img in images:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            resp = textract.analyze_document(Document={'Bytes': buf.getvalue()}, FeatureTypes=['FORMS'])
            blocks = resp.get('Blocks', [])

            block_map = {b['Id']: b for b in blocks}
            key_map = {b['Id']: b for b in blocks if b['BlockType']=='KEY_VALUE_SET' and 'KEY' in b.get('EntityTypes', [])}
            value_map = {b['Id']: b for b in blocks if b['BlockType']=='KEY_VALUE_SET' and 'VALUE' in b.get('EntityTypes', [])}

            def get_text(block):
                text = ""
                for rel in block.get('Relationships', []):
                    if rel['Type']=='CHILD':
                        for cid in rel['Ids']:
                            word = block_map.get(cid)
                            if word and word['BlockType']=='WORD':
                                text += word['Text'] + ' '
                return text.strip()

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

            for field, keywords in FIELD_KEYWORDS.items():
                for key_text, val_text in kvs.items():
                    if any(keyword in key_text for keyword in keywords):
                        results[field] = val_text
                        break
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI — File Upload & Extraction
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    f"Choose an image or PDF of a {document_type.lower()}",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded_file and openai.api_key and gemini_key:
    if st.button("🚀 Extract with GPT-4.1-mini, GPT-4.1, Gemini & AWS Textract", type="primary"):
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

        # Run LLM extractions
        try:
            dl_openai_mini = model_dl_from_images(b64_chunks, "gpt-4.1-mini")
        except Exception:
            dl_openai_mini = {k: "" for k in FIELDS}
        try:
            dl_gpt4_1 = model_dl_from_images(b64_chunks, "gpt-4.1")
        except Exception:
            dl_gpt4_1 = {k: "" for k in FIELDS}
        try:
            dl_gemini = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=4096
                ),
                contents=[types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png") for b64 in b64_chunks]
            )
            dl_gemini = json.loads(dl_gemini.text)
        except Exception:
            dl_gemini = {k: "" for k in FIELDS}
        try:
            dl_textract = textract_from_images(tmp_path)
        except Exception:
            dl_textract = {k: "" for k in FIELDS}

        st.success("Extraction complete!")

        # Display results
        col_img, col_models = st.columns([1, 2], gap="large")
        with col_img:
            st.subheader("🖼️ Converted Image(s)")
            for idx, img in enumerate(images, start=1):
                st.image(img, caption=f"Page {idx}", use_container_width=True)

        with col_models:
            tabs = st.tabs([f"{name} Fields" for name in ["GPT-4.1-mini", "GPT-4.1", "Gemini 2.5 Flash", "Textract"]])
            for tab, data in zip(tabs, [dl_openai_mini, dl_gpt4_1, dl_gemini, dl_textract]):
                with tab:
                    render_fields_grid(tab, f"{tab.title} Fields", data)

elif uploaded_file:
    st.info("Please provide OpenAI, Gemini, and AWS credentials to proceed.")
else:
    st.write("👈 Upload a file and provide API keys to get started.")
