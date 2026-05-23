"""
Streamlit Frontend for NLP IDU Project
=======================================
Interactive web interface for document analysis using the NLP IDU API.

Start with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import pdfplumber
import pandas as pd
import re
import json
import os
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Intelligent Document Understanding (IDU)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .entity-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin: 5px 0;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & HEADER
# ============================================================================
st.title("📄 Intelligent Document Understanding (IDU)")

st.markdown("""
**Advanced NLP Analysis Platform** - Extract insights from documents using state-of-the-art
natural language processing techniques.

---

### Key Features:
- 🏷️ **Document Classification** - Categorize documents automatically
- 🧠 **Named Entity Recognition** - Extract entities (names, organizations, locations, etc.)
- ✂️ **Text Summarization** - Generate concise summaries of long documents
- 📊 **Full Pipeline Analysis** - Run all analyses in one click
- 📥 **Multiple Input Methods** - Paste text or upload PDF/TXT files
""")

st.markdown("---")

# ============================================================================
# API CONFIGURATION  
# ============================================================================
API_BASE_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")  
API_ANALYZE_URL = f"{API_BASE_URL}/analyze"
API_HEALTH_URL = f"{API_BASE_URL}/health"

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Configuration")

# Task selection
task_option = st.sidebar.selectbox(
    "📋 Select Analysis Task",
    options=[
        "full_pipeline",
        "classification",
        "named_entity_recognition",
        "summarization"
    ],
    format_func=lambda x: {
        "full_pipeline": "🚀 Full Pipeline (All Tasks)",
        "classification": "🏷️ Classification Only",
        "named_entity_recognition": "🧠 NER Only",
        "summarization": "✂️ Summarization Only"
    }.get(x, x)
)

# Display options
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Display Options")

show_json = st.sidebar.checkbox("🧾 Show Raw JSON Response", value=False)
show_preview = st.sidebar.checkbox("👀 Show Text Preview", value=True)

# API Status
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 API Status")

try:
    # Changed from 5 to 15 to account for Render Free Tier spin-up latency
    health_response = requests.get(API_HEALTH_URL, timeout=15) 
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("status") == "healthy" or health_data.get("models_initialized"):
            st.sidebar.success("✅ API Connected & Ready (Lazy-loading enabled)")
        else:
            st.sidebar.warning("⚠️ API Online | Environment degraded...")
    else:
        st.sidebar.error("❌ API Error")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Cannot connect to API")
except Exception as e:
    st.sidebar.error(f"❌ Error: {str(e)[:50]}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file (TXT or PDF)."""
    try:
        if uploaded_file.type == "text/plain":
            return clean_text(uploaded_file.read().decode("utf-8"))
        
        elif uploaded_file.type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return clean_text(text)
        return None
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None


def get_text_statistics(text: str) -> dict:
    """Calculate text statistics."""
    words = text.split()
    sentences = [s for s in text.split(".") if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
    }


def display_entities(entities: list):
    """Display entities in a formatted way."""
    if not entities:
        st.info("ℹ️ No significant named entities detected.")
        return
    
    try:
        if isinstance(entities[0], dict):
            entity_df = pd.DataFrame(entities)
            # FIXED: Updated use_container_width=True -> width='stretch'
            st.dataframe(entity_df, width='stretch', hide_index=True)
        else:
            for idx, entity in enumerate(entities, 1):
                if isinstance(entity, dict):
                    st.markdown(f"**{idx}.** `{entity.get('text', 'N/A')}` 🏷️ **{entity.get('label', 'N/A')}**")
                else:
                    st.markdown(f"**{idx}.** {entity}")
    except Exception as e:
        for idx, entity in enumerate(entities, 1):
            st.text(f"{idx}. {entity}")


def display_entity_summary(entity_summary: dict):
    """Display entity summary statistics cleanly without crashing on overflows."""
    if not entity_summary:
        return
    
    cols = st.columns(3)
    for idx, (entity_type, count) in enumerate(entity_summary.items()):
        col_target = cols[idx % 3]
        with col_target:
            st.metric(label=f"Tag: {entity_type}", value=count)


# ============================================================================
# MAIN INPUT SECTION
# ============================================================================
st.subheader("📥 Input Document")

input_method = st.radio(
    "Choose how to provide your document:",
    options=["Paste Text", "Upload File"],
    horizontal=True,
    label_visibility="collapsed"
)

text = ""

if input_method == "Paste Text":
    text = st.text_area(
        label="📝 Enter or Paste Your Document",
        height=300,
        placeholder="Paste an article, report, business document, news article, research paper, etc...",
        label_visibility="collapsed"
    )
else:
    uploaded_file = st.file_uploader(
        "📤 Upload a TXT or PDF File",
        type=["txt", "pdf"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        with st.spinner("📖 Processing file..."):
            text = extract_text_from_file(uploaded_file)
        
        if text:
            st.success("✅ File processed successfully")
            if show_preview:
                with st.expander("👀 Preview Extracted Text", expanded=False):
                    st.text(text[:2000] + ("..." if len(text) > 2000 else ""))
        else:
            st.error("❌ Failed to extract text from file")

st.markdown("---")

col1, _, _ = st.columns([2, 1, 1])
with col1:
    # FIXED: Added explicit width alignment parameter
    analyze_button = st.button("🚀 Analyze Document", width='content', type="primary")

# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================
if analyze_button:
    if not text or len(text.strip()) < 50:
        st.warning("⚠️ Please provide at least 50 characters of text.")
    else:
        with st.spinner("🧠 Running NLP Pipeline... Note: First request could take a bit longer to lazily load model weights."):
            try:
                response = requests.post(
                    API_ANALYZE_URL,
                    json={"text": text, "task": task_option},
                    timeout=120
                )
                
                if response.status_code == 200:
                    api_response = response.json()
                    result = api_response.get("data", api_response)
                    
                    st.success("✅ Analysis Completed Successfully!")
                    st.markdown("---")
                    
                    # --- DOCUMENT STATISTICS ---
                    st.subheader("📊 Document Statistics")
                    stats = get_text_statistics(text)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Characters", f"{stats['characters']:,}")
                    c2.metric("Words", f"{stats['words']:,}")
                    c3.metric("Sentences", stats['sentences'])
                    c4.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
                    
                    st.markdown("---")
                    
                    # --- CLASSIFICATION RESULTS ---
                    if task_option in ["full_pipeline", "classification"]:
                        st.subheader("🏷️ Document Classification")
                        category = result.get("category") or api_response.get("category") or "Not classified"
                        st.info(f"**Predicted Category:** `{category}`")
                        st.markdown("---")
                    
                    # --- NAMED ENTITIES ---
                    if task_option in ["full_pipeline", "named_entity_recognition"]:
                        st.subheader("🧠 Named Entities Detected")
                        entities = result.get("entities") or api_response.get("entities", [])
                        if entities:
                            display_entities(entities)
                        else:
                            st.info("ℹ️ No significant named entities detected.")
                        
                        entity_summary = result.get("entity_summary") or api_response.get("entity_summary")
                        if entity_summary:
                            st.markdown("**Entity Summary Statistics:**")
                            display_entity_summary(entity_summary)
                        st.markdown("---")
                    
                    # --- SUMMARY ---
                    if task_option in ["full_pipeline", "summarization"]:
                        st.subheader("✂️ Generated Summary")
                        summary = result.get("summary") or api_response.get("summary") or "No summary available"
                        st.text_area(label="Summary", value=summary, height=200, disabled=True, label_visibility="collapsed")
                        
                        st.download_button(
                            label="⬇️ Download Summary as TXT",
                            data=summary,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        st.markdown("---")
                    
                    # --- RAW JSON ---
                    if show_json:
                        st.subheader("🧾 Raw API Response")
                        st.json(api_response)
                else:
                    st.error(f"❌ Backend Error (Status {response.status_code})")
                    try:
                        st.json(response.json())
                    except:
                        st.write(response.text)
                        
            except requests.exceptions.ConnectionError:
                st.error("❌ **Cannot connect to FastAPI backend.** Check if your API url parameter environment configurations are correct.")
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request Timeout.** The analysis task timeline expired.")
            except Exception as e:
                st.error(f"🚨 Unexpected Frontend Error: {str(e)}")

# ============================================================================
# FOOTER & INFORMATION
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📚 About
    **NLP IDU** is an intelligent document understanding system powered by:
    - spaCy for NER
    - Hugging Face Transformers for summarization
    - scikit-learn for classification
    """)
with col2:
    st.markdown("""
    ### 🚀 Getting Started
    1. Paste text or upload a file
    2. Select analysis type
    3. Click "Analyze Document"
    4. Download results
    """)
with col3:
    st.markdown("""
    ### 🔧 API Endpoints
    - `POST /analyze` - Single document
    - `POST /analyze-batch` - Multiple documents
    - `GET /health` - Health check
    - `GET /models/status` - Model status
                
    Full docs at `/docs`
    """)

st.markdown("---")
st.caption("🛠️ Built with Streamlit | FastAPI | spaCy | Transformers | scikit-learn")