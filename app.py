import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page configuration
st.set_page_config(
    page_title="Smart Resume Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Resume Assistant")
st.markdown("Analyze ATS compatibility, skill gaps, and keyword alignment using Machine Learning.")

# Helper function to extract text with OCR Fallback
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    
    # Method 1: Standard Text Extraction via pdfplumber
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0) # Reset stream pointer
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    except Exception as e:
        pass

    # Method 2: OCR Fallback via pytesseract if Method 1 returned no readable characters
    if len(extracted_text.strip()) == 0:
        st.info("ℹ️ Standard text layer not detected. Executing Optical Character Recognition (OCR)...")
        try:
            images = convert_from_bytes(pdf_bytes)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                extracted_text += ocr_text + "\n"
        except Exception as ocr_err:
            st.warning("OCR processing failed. Please re-export your PDF using MS Word or Google Docs.")

    return extracted_text.strip()

# Helper function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\-\/\s]', ' ', text)
    return " ".join(text.split())

# Main Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area("Paste Job Description (JD) here", height=200)

if st.button("Analyze Resume & Calculate Compatibility Score", type="primary"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Extracting text and calculating TF-IDF vector similarity..."):
            
            # Step 1: Extract and clean text
            resume_raw_text = extract_text_from_pdf(uploaded_file)
            
            if len(resume_raw_text.strip()) == 0:
                st.error("⚠️ Unable to parse this PDF. Please convert it to a standard PDF using Chrome or Google Docs.")
            else:
                resume_text = clean_text(resume_raw_text)
                jd_text = clean_text(job_description)
                
                # Debug Expander (Inspect extracted text)
                with st.expander("🔍 Inspect Extracted Raw Text"):
                    st.text_area("Parsed Resume Text", resume_raw_text, height=150)
                
                # Step 2: ML Model Training & Evaluation
                documents = [resume_text, jd_text]
                
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    token_pattern=r'(?u)\b[a-zA-Z0-9\+\#\-\/]{2,}\b',
                    ngram_range=(1, 1),
                    sublinear_tf=True
                )
                
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Calculate Similarity
                similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                match_score = round(similarity_matrix[0][0] * 100, 2)
                
                # Step 3: Skill Gap Analysis & Keyword Frequency Extraction
                feature_names = np.array(vectorizer.get_feature_names_out())
                resume_vector = tfidf_matrix[0].toarray()[0]
                jd_vector = tfidf_matrix[1].toarray()[0]
                
                df_words = pd.DataFrame({
                    'keyword': feature_names,
                    'resume_importance': resume_vector,
                    'jd_importance': jd_vector
                })
                
                # Keywords present in JD but missing/low in Resume
                missing_keywords = df_words[
                    (df_words['jd_importance'] > 0) & (df_words['resume_importance'] == 0)
                ].sort_values(by='jd_importance', ascending=False).head(10)
                
                # Top Matching Keywords
                matching_keywords = df_words[
                    (df_words['jd_importance'] > 0) & (df_words['resume_importance'] > 0)
                ].sort_values(by='jd_importance', ascending=False).head(10)
                
                # Display Results
                st.divider()
                st.header("📊 Evaluation Results")
                
                st.metric(label="Overall Candidate Compatibility Score", value=f"{match_score}%")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("🎯 Top Keyword Match")
                    if not matching_keywords.empty:
                        st.dataframe(
                            matching_keywords[['keyword']].reset_index(drop=True),
                            width="stretch"
                        )
                    else:
                        st.write("No direct keyword overlap detected.")

                with col_right:
                    st.subheader("⚠️ Missing Critical Keywords")
                    if not missing_keywords.empty:
                        st.dataframe(
                            missing_keywords[['keyword']].reset_index(drop=True),
                            width="stretch"
                        )
                    else:
                        st.write("No major skill gaps identified!")

                # Step 4: Seaborn Visualization
                st.divider()
                st.subheader("📈 Visualization: Keyword Importance Comparison (Seaborn)")
                
                top_jd_words = df_words.sort_values(by='jd_importance', ascending=False).head(8)
                plot_df = pd.melt(
                    top_jd_words, 
                    id_vars=['keyword'], 
                    value_vars=['resume_importance', 'jd_importance'],
                    var_name='Source', 
                    value_name='TF-IDF Score'
                )
                plot_df['Source'] = plot_df['Source'].map({
                    'resume_importance': 'Resume', 
                    'jd_importance': 'Job Description'
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(
                    data=plot_df, 
                    x='TF-IDF Score', 
                    y='keyword', 
                    hue='Source', 
                    palette=['#4C72B0', '#DD8452'],
                    ax=ax
                )
                ax.set_title("TF-IDF Vector Weight Comparison for Top JD Keywords")
                ax.set_xlabel("Feature Weight")
                ax.set_ylabel("Keyword")
                
                st.pyplot(fig)
            
    else:
        st.error("Please upload a PDF resume and provide a target job description.")