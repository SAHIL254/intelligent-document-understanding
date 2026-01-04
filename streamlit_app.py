import streamlit as st
import requests
import pdfplumber
import re

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Intelligent Document Understanding (IDU)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Understanding (IDU)")
st.write("Analyze documents using NLP: **Classification, NER & Summarization**")

API_URL = "http://127.0.0.1:8000/predict"

# ----------------------------
# Helper: Clean text
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# Helper: Read uploaded file
# ----------------------------
def extract_text(file):
    if file.type == "text/plain":
        return clean_text(file.read().decode("utf-8"))

    elif file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return clean_text(text)

    return None

# ----------------------------
# Input Section
# ----------------------------
st.subheader("📥 Input Document")

input_method = st.radio(
    "Choose input method:",
    ["Paste Text", "Upload File"],
    horizontal=True
)

text = ""

if input_method == "Paste Text":
    text = st.text_area(
        "Enter text here",
        height=220,
        placeholder="Paste a news article, report, or document text..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"]
    )

    if uploaded_file:
        text = extract_text(uploaded_file)
        if text:
            st.success("✅ File processed successfully")
        else:
            st.error("❌ Could not extract text")

# ----------------------------
# Submit
# ----------------------------
st.markdown("")

if st.button("🔍 Analyze Document", use_container_width=True):

    if not text or len(text) < 50:
        st.warning("⚠️ Please provide at least 50 characters of text")
    else:
        with st.spinner("🧠 Analyzing document..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("📊 Analysis Results")

                    # ----------------------------
                    # Category
                    # ----------------------------
                    st.markdown("### 🏷️ Document Category")
                    st.success(result.get("category", "Unknown"))

                    # ----------------------------
                    # Named Entities
                    # ----------------------------
                    st.markdown("### 🧠 Named Entities")

                    entities = result.get("entities", [])

                    if entities:
                        for ent, label in entities:
                            st.markdown(f"- **{ent}** 🏷️ `{label}`")
                    else:
                        st.info("No significant named entities detected.")

                    # ----------------------------
                    # Summary
                    # ----------------------------
                    st.markdown("### ✂️ Generated Summary")
                    st.text_area(
                        label="(You can copy and download this summary)",
                        value=result.get("summary", ""),
                        height=180
                    )

                    st.download_button(label="⬇️ Download Summary",
                                       data=result.get("summary", ""),
                                       file_name="summary.txt",
                                       mime="text/plain")


                else:
                    st.error("❌ Backend error. Check FastAPI logs.")

            except Exception as e:
                st.error(f"🚨 Error connecting to backend: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "Built with FastAPI, Streamlit, spaCy & HuggingFace Transformers • CPU-friendly NLP System"
)
