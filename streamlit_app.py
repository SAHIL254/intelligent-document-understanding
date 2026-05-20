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

API_BASE_URL = "http://127.0.0.1:8000"
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
    health_response = requests.get(API_HEALTH_URL, timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("models_loaded"):
            st.sidebar.success("✅ API Connected & Models Ready")
        else:
            st.sidebar.warning("⚠️ API Connected | Models Loading...")
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
        # TEXT FILE
        if uploaded_file.type == "text/plain":
            return clean_text(uploaded_file.read().decode("utf-8"))
        
        # PDF FILE
        elif uploaded_file.type == "application/pdf":
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return clean_text(text)
        
        else:
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
    
    # Try to display as DataFrame if possible
    try:
        if isinstance(entities[0], dict):
            entity_df = pd.DataFrame(entities)
            st.dataframe(entity_df, use_container_width=True, hide_index=True)
        else:
            # Display as list
            for idx, entity in enumerate(entities, 1):
                if isinstance(entity, dict):
                    st.markdown(
                        f"**{idx}.** `{entity.get('text', 'N/A')}` "
                        f"🏷️ **{entity.get('label', 'N/A')}**"
                    )
                else:
                    st.markdown(f"**{idx}.** {entity}")
    
    except Exception as e:
        # Fallback: simple display
        for idx, entity in enumerate(entities, 1):
            st.text(f"{idx}. {entity}")


def display_entity_summary(entity_summary: dict):
    """Display entity summary statistics."""
    if not entity_summary:
        return
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (entity_type, count) in enumerate(list(entity_summary.items())[:3]):
        with [col1, col2, col3][idx]:
            st.metric(
                label=f"{entity_type}",
                value=count,
                label_visibility="collapsed"
            )
    
    if len(entity_summary) > 3:
        with st.expander("📊 More Entity Types"):
            for entity_type, count in list(entity_summary.items())[3:]:
                st.metric(label=f"{entity_type}", value=count)


# ============================================================================
# MAIN INPUT SECTION
# ============================================================================

st.subheader("📥 Input Document")

# Input method selection
input_method = st.radio(
    "Choose how to provide your document:",
    options=["Paste Text", "Upload File"],
    horizontal=True,
    label_visibility="collapsed"
)

text = ""

# ============================================================================
# TEXT INPUT METHOD
# ============================================================================

if input_method == "Paste Text":
    
    text = st.text_area(
        label="📝 Enter or Paste Your Document",
        height=300,
        placeholder="Paste an article, report, business document, news article, research paper, etc...",
        label_visibility="collapsed"
    )

# ============================================================================
# FILE UPLOAD METHOD
# ============================================================================

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

# ============================================================================
# ANALYSIS BUTTON
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    analyze_button = st.button(
        "🚀 Analyze Document",
        width='content',
        type="primary"
    )

# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================

if analyze_button:
    
    # =====================================================================
    # INPUT VALIDATION
    # =====================================================================
    
    if not text or len(text.strip()) < 50:
        st.warning("⚠️ Please provide at least 50 characters of text.")
    
    else:
        
        # =====================================================================
        # API REQUEST
        # =====================================================================
        
        with st.spinner("🧠 Running NLP Pipeline... This may take a moment."):
            
            try:
                
                # Make API request
                response = requests.post(
                    API_ANALYZE_URL,
                    json={
                        "text": text,
                        "task": task_option
                    },
                    timeout=120
                )
                
                # =====================================================================
                # SUCCESS RESPONSE
                # =====================================================================
                
                if response.status_code == 200:
                    
                    api_response = response.json()
                    result = api_response.get("data", api_response)
                    
                    st.success("✅ Analysis Completed Successfully!")
                    
                    st.markdown("---")
                    
                    # =====================================================================
                    # DOCUMENT STATISTICS
                    # =====================================================================
                    
                    st.subheader("📊 Document Statistics")
                    
                    stats = get_text_statistics(text)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Characters", f"{stats['characters']:,}")
                    col2.metric("Words", f"{stats['words']:,}")
                    col3.metric("Sentences", stats['sentences'])
                    col4.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
                    
                    st.markdown("---")
                    
                    # =====================================================================
                    # CLASSIFICATION RESULTS
                    # =====================================================================
                    
                    if task_option in ["full_pipeline", "classification"]:
                        
                        st.subheader("🏷️ Document Classification")
                        
                        category = (
                            result.get("category")
                            or api_response.get("category")
                            or "Not classified"
                        )
                        
                        st.info(f"**Predicted Category:** `{category}`")
                        
                        st.markdown("---")
                    
                    # =====================================================================
                    # NAMED ENTITIES
                    # =====================================================================
                    
                    if task_option in ["full_pipeline", "named_entity_recognition"]:
                        
                        st.subheader("🧠 Named Entities Detected")
                        
                        entities = (
                            result.get("entities")
                            or api_response.get("entities", [])
                        )
                        
                        if entities:
                            display_entities(entities)
                        else:
                            st.info("ℹ️ No significant named entities detected.")
                        
                        # Entity Summary
                        entity_summary = (
                            result.get("entity_summary")
                            or api_response.get("entity_summary")
                        )
                        
                        if entity_summary:
                            st.markdown("**Entity Summary:**")
                            display_entity_summary(entity_summary)
                        
                        st.markdown("---")
                    
                    # =====================================================================
                    # SUMMARY
                    # =====================================================================
                    
                    if task_option in ["full_pipeline", "summarization"]:
                        
                        st.subheader("✂️ Generated Summary")
                        
                        summary = (
                            result.get("summary")
                            or api_response.get("summary")
                            or "No summary available"
                        )
                        
                        st.text_area(
                            label="Summary",
                            value=summary,
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download Summary as TXT",
                            data=summary,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                        st.markdown("---")
                    
                    # =====================================================================
                    # RAW JSON (Optional)
                    # =====================================================================
                    
                    if show_json:
                        
                        st.subheader("🧾 Raw API Response")
                        
                        st.json(api_response)
                
                # =====================================================================
                # ERROR RESPONSE
                # =====================================================================
                
                else:
                    
                    st.error(f"❌ Backend Error (Status {response.status_code})")
                    
                    try:
                        error_data = response.json()
                        st.json(error_data)
                    except:
                        st.write(response.text)
            
            # =====================================================================
            # CONNECTION ERROR
            # =====================================================================
            
            except requests.exceptions.ConnectionError:
                
                st.error("""
                    ❌ **Cannot connect to FastAPI backend**
                    
                    Please start the backend server first:
                    
                    ```bash
                    cd /path/to/nlp-idu
                    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
                    ```
                    
                    Or if using a remote server, update the API_BASE_URL in the app.
                """)
            
            # =====================================================================
            # TIMEOUT ERROR
            # =====================================================================
            
            except requests.exceptions.Timeout:
                
                st.error("""
                    ⏱️ **Request Timeout**
                    
                    The analysis took too long to complete.
                    Try with a shorter document or increase the timeout.
                """)
            
            # =====================================================================
            # GENERAL ERROR
            # =====================================================================
            
            except Exception as e:
                
                st.error(f"🚨 Unexpected Error: {str(e)}")
                
                with st.expander("Error Details"):
                    st.text(str(e))

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
    
    **Need help?** Check the documentation.
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

st.caption(
    "🛠️ Built with Streamlit | FastAPI | spaCy | Transformers | scikit-learn"
)