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
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for UI Badges and Hero Cards
st.markdown("""
<style>
    .hero-card {
        background-color: #1E1E2E;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #313244;
    }
    .hero-score {
        font-size: 54px;
        font-weight: 800;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin-top: 10px;
    }
    .badge-matched {
        background-color: #1E3A2B;
        color: #2ECC71;
        border: 1px solid #2ECC71;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
    .badge-missing {
        background-color: #3A1E1E;
        color: #E74C3C;
        border: 1px solid #E74C3C;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        margin: 4px;
        font-size: 13px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Smart Resume Assistant & ATS Optimizer")
st.caption("AI-powered candidate-job fit analyzer with technical skill taxonomy & LLM recommendations.")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")

# Predefined Technical Taxonomy
TECH_TAXONOMY = {
    'python', 'java', 'c', 'cpp', 'c++', 'c#', 'javascript', 'typescript', 'go', 'golang', 'rust', 'sql', 'bash', 'powershell', 'html', 'css',
    'spring boot', 'spring', 'react', 'react.js', 'next.js', 'flask', 'django', 'streamlit', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'pandas', 'numpy', 'seaborn', 'matplotlib',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'cloudformation', 'bitbucket', 'gitlab', 'git', 'ci/cd', 'ec2', 'ecs', 's3', 'lambda', 'rds', 'kms', 'elb', 'sns', 'sqs', 'iam',
    'microservices', 'rest api', 'graphql', 'linux', 'networking', 'firewalls', 'routing', 'agile', 'data pipelines', 'data warehouse', 'data lakes', 'dbms', 'oop', 'machine learning', 'deep learning', 'cnn-lstm', 'dqn'
}

def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text.strip()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\-\/\s]', ' ', text)
    return " ".join(text.split())

def extract_tech_skills(text):
    cleaned = clean_text(text)
    found_skills = set()
    for skill in TECH_TAXONOMY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, cleaned):
            found_skills.add(skill)
    return found_skills

# Upload Section
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("1. Upload Candidate Resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area("2. Paste Job Description (JD)", height=160)

if st.button("🚀 Analyze Compatibility", type="primary", use_container_width=True):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Analyzing resume against job parameters..."):
            
            resume_raw_text = extract_text_from_pdf(uploaded_file)
            
            if len(resume_raw_text.strip()) == 0:
                st.error("⚠️ Unable to extract text from PDF.")
            else:
                resume_text = clean_text(resume_raw_text)
                jd_text = clean_text(job_description)
                
                # TF-IDF Vector Similarity
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    token_pattern=r'(?u)\b[a-zA-Z0-9\+\#\-\/]{2,}\b',
                    ngram_range=(1, 1),
                    sublinear_tf=True
                )
                tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
                raw_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
                
                # Tech Skill Extraction
                resume_tech = extract_tech_skills(resume_raw_text)
                jd_tech = extract_tech_skills(job_description)
                
                matching_tech = resume_tech.intersection(jd_tech)
                missing_tech = jd_tech - resume_tech
                
                tech_match_score = (len(matching_tech) / len(jd_tech) * 100) if len(jd_tech) > 0 else raw_similarity
                final_ats_score = round((0.40 * raw_similarity) + (0.60 * tech_match_score), 1)
                
                # Color and Status Formatting
                if final_ats_score >= 70:
                    score_color = "#2ECC71"
                    status_text = "🟢 Strong Match — Ready to Apply"
                    badge_bg = "#1E3A2B"
                elif final_ats_score >= 45:
                    score_color = "#F39C12"
                    status_text = "🟡 Moderate Match — Needs Minor Adjustments"
                    badge_bg = "#3A301E"
                else:
                    score_color = "#E74C3C"
                    status_text = "🔴 Needs Tailoring — Significant Skill Gap"
                    badge_bg = "#3A1E1E"

                st.divider()

                # Hero Score Header
                st.markdown(f"""
                <div class="hero-card">
                    <div style="color: #A6ADC8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Overall ATS Match Score</div>
                    <div class="hero-score" style="color: {score_color};">{final_ats_score}%</div>
                    <div class="status-badge" style="background-color: {badge_bg}; color: {score_color}; border: 1px solid {score_color};">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sub-Score Breakdown Expander
                with st.expander("🔍 View Technical Score Breakdown"):
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Tech Stack Match Rate", f"{round(tech_match_score, 1)}%", help="Percentage of required technical skills present on your resume.")
                    sc2.metric("Contextual Vector Score", f"{round(raw_similarity, 1)}%", help="TF-IDF keyword density similarity.")

                # Tabbed Interface
                tab1, tab2, tab3 = st.tabs(["🛠️ Skill Gap Analysis", "📈 Breakdown Chart", "🤖 AI Strategy Coach"])

                with tab1:
                    t_col1, t_col2 = st.columns(2)
                    
                    with t_col1:
                        st.markdown(f"#### 🎯 Matching Tech Skills ({len(matching_tech)})")
                        if matching_tech:
                            badges_html = "".join([f'<span class="badge-matched">✓ {skill.upper()}</span>' for skill in sorted(list(matching_tech))])
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.info("No direct technical skills matched.")

                    with t_col2:
                        st.markdown(f"#### ⚠️ Missing Tech Skills ({len(missing_tech)})")
                        if missing_tech:
                            badges_html = "".join([f'<span class="badge-missing">✕ {skill.upper()}</span>' for skill in sorted(list(missing_tech))])
                            st.markdown(badges_html, unsafe_allow_html=True)
                        else:
                            st.success("All required technical skills are covered!")

                with tab2:
                    tech_summary_df = pd.DataFrame({
                        'Category': ['Matching Tech Skills', 'Missing Tech Skills'],
                        'Count': [len(matching_tech), len(missing_tech)]
                    })
                    fig, ax = plt.subplots(figsize=(7, 2.5))
                    sns.barplot(data=tech_summary_df, x='Count', y='Category', palette=['#2ECC71', '#E74C3C'], ax=ax)
                    ax.set_title("Tech Stack Coverage Ratio", fontsize=11)
                    ax.set_xlabel("Number of Skills")
                    ax.set_ylabel("")
                    st.pyplot(fig)

                with tab3:
                    if gemini_api_key:
                        with st.spinner("Consulting Gemini AI Coach..."):
                            try:
                                client = genai.Client(api_key=gemini_api_key)
                                prompt = f"""
                                You are an executive ATS Coach. Candidate scored {final_ats_score}% for this role.
                                Missing Skills: {', '.join(missing_tech)}
                                Job Description: {job_description}
                                Candidate Resume: {resume_raw_text}
                                
                                Provide 3 bullet-point recommendations to seamlessly incorporate the top missing tech skills.
                                """
                                response = client.models.generate_content(
                                    model='gemini-3.8-flash',
                                    contents=prompt
                                )
                                st.markdown(response.text)
                            except Exception as ai_err:
                                st.error(f"Gemini API Error: {ai_err}")
                    else:
                        st.info("💡 Add a free Gemini API key in the sidebar to get instant AI-generated bullet rewrite suggestions.")

                # Downloadable Audit Report
                st.divider()
                report_content = f"""ATS RESUME EVALUATION REPORT
===========================
Overall ATS Score: {final_ats_score}%
Status: {status_text}

TECH STACK MATCHES ({len(matching_tech)}):
{', '.join(sorted(list(matching_tech)))}

MISSING TECH SKILLS ({len(missing_tech)}):
{', '.join(sorted(list(missing_tech)))}
"""
                st.download_button(
                    label="📥 Download Summary Report (.txt)",
                    data=report_content,
                    file_name="ATS_Resume_Evaluation_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    else:
        st.error("Please upload a PDF resume and provide a target job description.")