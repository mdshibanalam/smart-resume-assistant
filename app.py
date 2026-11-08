import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from google import genai

# Page configuration
st.set_page_config(
    page_title="Smart Resume Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Resume Assistant & ATS Optimizer")
st.markdown("Analyze ATS compatibility, skill gaps, and keyword alignment using Machine Learning & Generative AI.")

# Sidebar for Gemini API Key
st.sidebar.header("🔑 AI Integration Settings")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key (Optional)", type="password")
st.sidebar.info("Get a free key from Google AI Studio to unlock LLM-powered bullet rewrites and strategic advice.")

# Helper function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text.strip()

# Helper function to clean text while preserving technical symbols
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
                st.error("⚠️ Unable to extract text from this PDF. Please check file encoding.")
            else:
                resume_text = clean_text(resume_raw_text)
                jd_text = clean_text(job_description)
                
                # Debug Expander
                with st.expander("🔍 Inspect Extracted Raw Text"):
                    st.text_area("Parsed Resume Text", resume_raw_text, height=150)
                
                # Step 2: TF-IDF Vectorizer with Custom Stop Words Filter
                base_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())
                domain_noise = [
                    'work', 'salary', 'knowledge', 'barclays', 'details', 'bengaluru', 
                    'services', 'collaborate', 'build', 'candidate', 'candidates', 'ability',
                    'looking', 'role', 'roles', 'responsibilities', 'qualifications', 'environment',
                    'implement', 'understanding', 'lpa', 'engineer', '2023', '2024', '2022', '2025', '2026'
                ]
                custom_stop_words = list(set(base_stop_words + domain_noise))
                
                vectorizer = TfidfVectorizer(
                    stop_words=custom_stop_words,
                    token_pattern=r'(?u)\b[a-zA-Z0-9\+\#\-\/]{2,}\b',
                    ngram_range=(1, 1),
                    sublinear_tf=True
                )
                
                documents = [resume_text, jd_text]
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
                
                # Filter out pure numbers from keywords
                df_words = df_words[~df_words['keyword'].str.isnumeric()]
                
                missing_keywords = df_words[
                    (df_words['jd_importance'] > 0) & (df_words['resume_importance'] == 0)
                ].sort_values(by='jd_importance', ascending=False).head(10)
                
                matching_keywords = df_words[
                    (df_words['jd_importance'] > 0) & (df_words['resume_importance'] > 0)
                ].sort_values(by='jd_importance', ascending=False).head(10)
                
                # Display Statistical Results
                st.divider()
                st.header("📊 Evaluation Results")
                st.metric(label="Overall Candidate Compatibility Score", value=f"{match_score}%")
                
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("🎯 Top Keyword Match")
                    if not matching_keywords.empty:
                        st.dataframe(matching_keywords[['keyword']].reset_index(drop=True), width="stretch")
                    else:
                        st.write("No direct keyword overlap detected.")

                with col_right:
                    st.subheader("⚠️ Missing Critical Technical Keywords")
                    if not missing_keywords.empty:
                        st.dataframe(missing_keywords[['keyword']].reset_index(drop=True), width="stretch")
                    else:
                        st.write("No major skill gaps identified!")

                # Seaborn Visualization
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
                plot_df['Source'] = plot_df['Source'].map({'resume_importance': 'Resume', 'jd_importance': 'Job Description'})
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=plot_df, x='TF-IDF Score', y='keyword', hue='Source', palette=['#4C72B0', '#DD8452'], ax=ax)
                ax.set_title("TF-IDF Vector Weight Comparison for Top Technical Keywords")
                st.pyplot(fig)

                # Step 4: Optional Gemini API AI Strategic Advice Module
                if gemini_api_key:
                    st.divider()
                    st.header("🤖 Gemini AI Strategic Resume Optimization")
                    with st.spinner("Generating LLM strategic recommendations..."):
                        try:
                            client = genai.Client(api_key=gemini_api_key)
                            prompt = f"""
                            You are an expert ATS Resume Coach. Analyze the candidate resume against the Job Description.
                            
                            Job Description:
                            {job_description}
                            
                            Candidate Resume:
                            {resume_raw_text}
                            
                            Provide:
                            1. Three critical missing technical skills or concepts that should be added.
                            2. Two specific bullet points from the candidate's resume rewritten to better match the JD key requirements.
                            3. Final actionable recommendation to increase ATS score.
                            """
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=prompt
                            )
                            st.markdown(response.text)
                        except Exception as ai_err:
                            st.error(f"Gemini API Error: {ai_err}")

    else:
        st.error("Please upload a PDF resume and provide a target job description.")