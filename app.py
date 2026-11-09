import io
import os
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SMARA - Smart Resume Assistant",
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

# App Branding Header
st.title("⚡ SMARA - Smart Resume Assistant")
st.caption("An Intelligent Applicant Tracking System with An AI-Powered Resume Tailor Assistant - Powered by SMARA AI")

# Sidebar Configuration & Early API Authentication
st.sidebar.header("⚙️ AI Configuration")
env_api_key = os.getenv("GEMINI_API_KEY", "")

if env_api_key:
    user_api_key = env_api_key
else:
    user_api_key = st.sidebar.text_input("Gemini API Key (Manual Override)", type="password")
    if not user_api_key:
        st.sidebar.info("💡 Add `GEMINI_API_KEY` to your `.env` file to enable SMARA AI automatically.")

# Perform Early API Authentication Verification
gemini_client = None
api_valid = False

if user_api_key:
    try:
        test_client = genai.Client(api_key=user_api_key)
        # Perform lightweight check call to verify credentials on app load
        test_client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents='Ping'
        )
        st.sidebar.success("SMARA AI Enabled ✅")
        gemini_client = test_client
        api_valid = True
    except Exception as auth_error:
        # Simplify common exception messages for clean sidebar display
        err_msg = str(auth_error)
        if "API_KEY_INVALID" in err_msg or "400" in err_msg or "403" in err_msg:
            st.sidebar.error("❌ SMARA AI Auth Failed: Invalid API Key")
        else:
            st.sidebar.error(f"❌ SMARA AI Auth Failed: {err_msg[:80]}")
        api_valid = False
        gemini_client = None

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

# Calculate and store results in Session State upon button click
if st.button("🚀 Analyze Compatibility", type="primary", use_container_width=True):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("SMARA is analyzing candidate parameters..."):
            
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

                # Generate Graph Image Buffer for Download in Tab 2
                tech_summary_df = pd.DataFrame({
                    'Category': ['Matching Tech Skills', 'Missing Tech Skills'],
                    'Count': [len(matching_tech), len(missing_tech)]
                })
                fig, ax = plt.subplots(figsize=(7, 2.5))
                sns.barplot(data=tech_summary_df, x='Count', y='Category', palette=['#2ECC71', '#E74C3C'], ax=ax)
                ax.set_title("Tech Stack Coverage Ratio", fontsize=11)
                ax.set_xlabel("Number of Skills")
                ax.set_ylabel("")
                
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=300)
                img_buf.seek(0)

                # Generate Summary Report text for Tab 1
                smara_report_text = f"""SMARA REPORT
===========================
Overall ATS Score: {final_ats_score}%
Status: {status_text}

TECH STACK MATCHES ({len(matching_tech)}):
{', '.join(sorted(list(matching_tech))) if matching_tech else 'None'}

MISSING TECH SKILLS ({len(missing_tech)}):
{', '.join(sorted(list(missing_tech))) if missing_tech else 'None'}
"""

                # Generate SMARA AI Report if authenticated
                ai_report_text = None
                ai_recommendation_text = None
                
                if api_valid and gemini_client:
                    try:
                        prompt = f"""
                        You are SMARA AI, an executive ATS Resume Coach. Candidate scored {final_ats_score}% for this role.
                        Missing Skills: {', '.join(missing_tech)}
                        Job Description: {job_description}
                        Candidate Resume: {resume_raw_text}
                        
                        Provide 3 bullet-point recommendations to seamlessly incorporate the top missing tech skills into candidate project descriptions.
                        """
                        response = gemini_client.models.generate_content(
                            model='gemini-flash-lite-latest',
                            contents=prompt
                        )
                        ai_recommendation_text = response.text
                        ai_report_text = f"""===================================
SMARA AI REPORT
===================================
Candidate ATS Score: {final_ats_score}%
Status: {status_text}

MISSING TECH SKILLS:
{', '.join(sorted(list(missing_tech))) if missing_tech else 'None'}

SMARA AI STRATEGIC RECOMMENDATIONS:
-----------------------------------
{ai_recommendation_text}
"""
                    except Exception as ai_exec_err:
                        ai_recommendation_text = f"Error generating recommendations: {ai_exec_err}"

                # Save all calculated variables to session state
                st.session_state['results'] = {
                    'final_ats_score': final_ats_score,
                    'tech_match_score': tech_match_score,
                    'raw_similarity': raw_similarity,
                    'matching_tech': matching_tech,
                    'missing_tech': missing_tech,
                    'score_color': score_color,
                    'status_text': status_text,
                    'badge_bg': badge_bg,
                    'fig': fig,
                    'graph_png_bytes': img_buf.getvalue(),
                    'smara_report_text': smara_report_text,
                    'ai_report_text': ai_report_text,
                    'ai_recommendation_text': ai_recommendation_text
                }
    else:
        st.error("Please upload a PDF resume and provide a target job description.")

# Render UI from Session State to ensure download buttons do NOT refresh/wipe the dashboard
if 'results' in st.session_state:
    res = st.session_state['results']
    
    st.divider()

    # Hero Score Header
    st.markdown(f"""
    <div class="hero-card">
        <div style="color: #A6ADC8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">SMARA Overall ATS Match Score</div>
        <div class="hero-score" style="color: {res['score_color']};">{res['final_ats_score']}%</div>
        <div class="status-badge" style="background-color: {res['badge_bg']}; color: {res['score_color']}; border: 1px solid {res['score_color']};">
            {res['status_text']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sub-Score Breakdown Expander
    with st.expander("🔍 View Technical Score Breakdown"):
        sc1, sc2 = st.columns(2)
        sc1.metric("Tech Stack Match Rate", f"{round(res['tech_match_score'], 1)}%", help="Percentage of required technical skills present on your resume.")
        sc2.metric("Contextual Vector Score", f"{round(res['raw_similarity'], 1)}%", help="TF-IDF keyword density similarity.")

    # Tabbed Interface
    tab1, tab2, tab3 = st.tabs(["🛠️ Skill Gap Analysis", "📈 Breakdown Chart", "🤖 SMARA AI Coach"])

    # TAB 1: Skill Gap Analysis & SMARA Report Download
    with tab1:
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.markdown(f"#### 🎯 Matching Tech Skills ({len(res['matching_tech'])})")
            if res['matching_tech']:
                badges_html = "".join([f'<span class="badge-matched">✓ {skill.upper()}</span>' for skill in sorted(list(res['matching_tech']))])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("No direct technical skills matched.")

        with t_col2:
            st.markdown(f"#### ⚠️ Missing Tech Skills ({len(res['missing_tech'])})")
            if res['missing_tech']:
                badges_html = "".join([f'<span class="badge-missing">✕ {skill.upper()}</span>' for skill in sorted(list(res['missing_tech']))])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.success("All required technical skills are covered!")

        st.divider()
        st.download_button(
            label="📥 Download SMARA Report (.txt)",
            data=res['smara_report_text'],
            file_name="SMARA_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # TAB 2: Breakdown Chart & SMARA Graph Download
    with tab2:
        st.pyplot(res['fig'])
        st.divider()
        st.download_button(
            label="📥 Download SMARA Graph (.png)",
            data=res['graph_png_bytes'],
            file_name="SMARA_Graph.png",
            mime="image/png",
            use_container_width=True
        )

    # TAB 3: SMARA AI Coach & SMARA AI Report Download
    with tab3:
        st.markdown("### 🤖 SMARA AI Report")
        if api_valid and res['ai_recommendation_text']:
            st.markdown(res['ai_recommendation_text'])
            st.divider()
            if res['ai_report_text']:
                st.download_button(
                    label="📥 Download SMARA AI Report (.txt)",
                    data=res['ai_report_text'],
                    file_name="SMARA_AI_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            if not user_api_key:
                st.info("💡 API Key missing. Add `GEMINI_API_KEY` to `.env` or input it in the sidebar to generate the SMARA AI Report.")
            else:
                st.error("❌ Authentication failed for Gemini API Key. Please verify key credentials in the sidebar.")