import streamlit as st
import requests
import pdfplumber
import pandas as pd
import re
import json

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Intelligent Document Understanding (IDU)",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📄 Intelligent Document Understanding (IDU)")

st.markdown("""
Analyze documents using advanced NLP techniques:

- 🏷️ Document Classification
- 🧠 Named Entity Recognition (NER)
- ✂️ Text Summarization
- 📊 Full NLP Pipeline Analysis
""")

# ---------------------------------------------------
# FASTAPI ENDPOINT
# ---------------------------------------------------
API_URL = "http://127.0.0.1:8000/analyze"

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Analysis Settings")

task_option = st.sidebar.selectbox(
    "Select NLP Task",
    [
        "full_pipeline",
        "classification",
        "named_entity_recognition",
        "summarization"
    ]
)

show_json = st.sidebar.checkbox("Show Raw JSON Output")

# ---------------------------------------------------
# CLEAN TEXT FUNCTION
# ---------------------------------------------------
def clean_text(text: str):

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------------------------------
# FILE TEXT EXTRACTION
# ---------------------------------------------------
def extract_text(file):

    # TEXT FILE
    if file.type == "text/plain":

        return clean_text(
            file.read().decode("utf-8")
        )

    # PDF FILE
    elif file.type == "application/pdf":

        text = ""

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + " "

        return clean_text(text)

    return None

# ---------------------------------------------------
# TEXT STATISTICS
# ---------------------------------------------------
def get_text_statistics(text):

    words = text.split()

    sentences = text.split(".")

    sentence_count = len(
        [s for s in sentences if s.strip()]
    )

    return {
        "Characters": len(text),
        "Words": len(words),
        "Sentences": sentence_count
    }

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
st.subheader("📥 Input Document")

input_method = st.radio(
    "Choose Input Method",
    ["Paste Text", "Upload File"],
    horizontal=True
)

text = ""

# ---------------------------------------------------
# PASTE TEXT
# ---------------------------------------------------
if input_method == "Paste Text":

    text = st.text_area(
        "Enter Document Text",
        height=250,
        placeholder="Paste article, report, business document, news article, etc..."
    )

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF File",
        type=["txt", "pdf"]
    )

    if uploaded_file:

        text = extract_text(uploaded_file)

        if text:

            st.success("✅ File processed successfully")

            with st.expander("📄 Preview Extracted Text"):

                st.write(text[:3000])

        else:

            st.error("❌ Failed to extract text")

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------
st.markdown("")

if st.button("🚀 Analyze Document", use_container_width=True):

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------
    if not text or len(text) < 50:

        st.warning("⚠️ Please provide at least 50 characters of text.")

    else:

        with st.spinner("🧠 Running NLP Pipeline..."):

            try:

                # ---------------------------------------------------
                # API REQUEST
                # ---------------------------------------------------
                response = requests.post(
                    API_URL,
                    json={
                        "text": text,
                        "task": task_option
                    },
                    timeout=120
                )

                # ---------------------------------------------------
                # SUCCESS
                # ---------------------------------------------------
                if response.status_code == 200:

                    api_response = response.json()

                    # IMPORTANT
                    result = api_response.get("data", {})

                    st.success("✅ Analysis Completed Successfully")

                    # ---------------------------------------------------
                    # DOCUMENT STATISTICS
                    # ---------------------------------------------------
                    st.subheader("📊 Document Statistics")

                    stats = get_text_statistics(text)

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Characters",
                        stats["Characters"]
                    )

                    col2.metric(
                        "Words",
                        stats["Words"]
                    )

                    col3.metric(
                        "Sentences",
                        stats["Sentences"]
                    )

                    st.markdown("---")

                    # ---------------------------------------------------
                    # DOCUMENT CATEGORY
                    # ---------------------------------------------------
                    st.subheader("🏷️ Document Classification")

                    # your original logic + robust extraction
                    category = (
                        result.get("category")
                        or api_response.get("category")
                        or "Unknown"
                    )

                    st.success(f"Predicted Category: {category}")

                    # ---------------------------------------------------
                    # NAMED ENTITIES
                    # ---------------------------------------------------
                    st.subheader("🧠 Named Entities")

                    entities = (
                        result.get("entities")
                        or api_response.get("entities", [])
                    )

                    if entities:

                        # Case 1 -> Dictionary entities
                        if isinstance(entities[0], dict):

                            entity_df = pd.DataFrame(entities)

                            st.dataframe(
                                entity_df,
                                use_container_width=True
                            )

                        # Case 2 -> Tuple/List entities
                        else:

                            for ent, label in entities:

                                st.markdown(
                                    f"- **{ent}** 🏷️ `{label}`"
                                )

                    else:

                        st.info(
                            "No significant named entities detected."
                        )

                    # ---------------------------------------------------
                    # ENTITY SUMMARY
                    # ---------------------------------------------------
                    entity_summary = (
                        result.get("entity_summary")
                        or api_response.get("entity_summary")
                    )

                    if entity_summary:

                        st.subheader("📊 Entity Groups")

                        st.json(entity_summary)

                    # ---------------------------------------------------
                    # GENERATED SUMMARY
                    # ---------------------------------------------------
                    st.subheader("✂️ Generated Summary")

                    summary = (
                        result.get("summary")
                        or api_response.get("summary")
                        or "No summary generated."
                    )

                    st.text_area(
                        "Summary",
                        value=summary,
                        height=220
                    )

                    # ---------------------------------------------------
                    # DOWNLOAD SUMMARY
                    # ---------------------------------------------------
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

                    # ---------------------------------------------------
                    # RAW JSON
                    # ---------------------------------------------------
                    if show_json:

                        st.subheader("🧾 Raw API Response")

                        st.json(api_response)

                # ---------------------------------------------------
                # API ERROR
                # ---------------------------------------------------
                else:

                    st.error(
                        f"❌ Backend Error ({response.status_code})"
                    )

                    try:

                        st.json(response.json())

                    except:

                        st.write(response.text)

            # ---------------------------------------------------
            # CONNECTION ERROR
            # ---------------------------------------------------
            except requests.exceptions.ConnectionError:
                st.error("""
                        ❌ Unable to connect to FastAPI backend.
                        Start backend server first:

                        ```bash
                        uvicorn app.main:app --reload
                                                                """)
            except Exception as e:
                st.error(f"🚨 Unexpected Error: {e}")

#---------------------------------------------------
#FOOTER
#---------------------------------------------------

st.markdown("---")

st.caption(
"Built with Streamlit, FastAPI, spaCy & HuggingFace Transformers"
)